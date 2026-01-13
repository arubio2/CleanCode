import os
import re
import json
import argparse
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import PP_PLACEHOLDER
from pptx.enum.text import PP_ALIGN
from openai import OpenAI

def delete_slide(presentation, index):
    """Deletes a slide from the presentation by its index."""
    xml_slides = presentation.slides._sldIdLst
    slides = list(xml_slides)
    xml_slides.remove(slides[index])

def parse_markdown(md_content):
    """Extracts text and identifies image paths found in the report."""
    images = re.findall(r'!\[.*?\]\((.*?)\)', md_content)
    return md_content, images

def get_template_layouts(pptx_path):
    """Extracts detailed metadata from the template to help the LLM select layouts."""
    prs = Presentation(pptx_path)
    layouts = []
    for i, layout in enumerate(prs.slide_layouts):
        placeholders = []
        for p in layout.placeholders:
            ph_info = {
                "name": p.name,
                "type": str(p.placeholder_format.type),
                "left": p.left,
                "top": p.top,
                "width": p.width,
                "height": p.height
            }
            placeholders.append(ph_info)
        
        layouts.append({
            "index": i,
            "name": layout.name,
            "placeholders": placeholders,
            "placeholder_count": len(placeholders)
        })
    return layouts

def auto_fit_text(text_frame, max_font_size=18, min_font_size=10):
    """Automatically adjusts font size to fit content in text frame."""
    if not text_frame.text.strip():
        return
    
    # Start with max font size and reduce if needed
    for size in range(max_font_size, min_font_size - 1, -1):
        for paragraph in text_frame.paragraphs:
            for run in paragraph.runs:
                run.font.size = Pt(size)
        
        # Check if text fits (this is approximate)
        if text_frame.text:
            break

def summarize_and_map_with_llm(md_content, layouts, image_list, api_key):
    """Consults the LLM to structure the report and pick the best template layouts."""
    client = OpenAI(api_key=api_key)
    
    # Create more detailed layout descriptions
    layout_descriptions = []
    for l in layouts:
        ph_types = [p['type'] for p in l['placeholders']]
        has_picture = any('PICTURE' in t for t in ph_types)
        has_body = any('BODY' in t or 'OBJECT' in t for t in ph_types)
        
        desc = f"- Index {l['index']}: '{l['name']}' ({l['placeholder_count']} placeholders)"
        if has_picture:
            desc += " [SUPPORTS IMAGES]"
        if has_body:
            desc += " [HAS TEXT BODY]"
        layout_descriptions.append(desc)
    
    layout_context = "\n".join(layout_descriptions)
    
    prompt = f"""
    You are a professional presentation designer. Convert the following report into a slide deck.
    
    AVAILABLE TEMPLATE LAYOUTS:
    {layout_context}
    
    AVAILABLE IMAGES:
    {", ".join(image_list) if image_list else "No images available"}
    
    CRITICAL INSTRUCTIONS FOR TEXT FITTING:
    1. Create a title slide first (usually layout 0 or one named 'Title Slide')
    2. **MAXIMUM 4 bullets per slide** - This is a hard limit for readability
    3. **Each bullet MUST be under 50 characters** - Longer text won't fit properly
    4. **If a topic needs more than 4 bullets, split into multiple slides** with titles like:
       - "Topic Name (1/2)" and "Topic Name (2/2)" 
       - "Topic Name - Part 1" and "Topic Name - Part 2"
    5. For slides with images, prefer layouts marked [SUPPORTS IMAGES]
    6. For text-heavy slides, use layouts with [HAS TEXT BODY]
    7. Better to have more slides with less content than fewer slides that are overcrowded
    
    RETURN FORMAT:
    Return a JSON object with key 'slides', where each slide has:
    - "title": slide title (keep under 45 characters)
    - "bullets": list of bullet points (MAX 4 items, each under 50 chars)
    - "image_path": path from available images or null
    - "layout_index": best matching layout index
    - "notes": any additional context (optional)
    
    REPORT:
    {md_content}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a PowerPoint automation expert. Output valid JSON only. ALWAYS split content into multiple slides if needed. Never exceed 4 bullets per slide or 50 characters per bullet."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content).get('slides', [])

def fit_image_to_placeholder(placeholder, image_path):
    """Inserts image while maintaining aspect ratio within placeholder bounds."""
    from PIL import Image
    
    # Get placeholder dimensions
    ph_width = placeholder.width
    ph_height = placeholder.height
    ph_left = placeholder.left
    ph_top = placeholder.top
    
    # Get image dimensions
    with Image.open(image_path) as img:
        img_width, img_height = img.size
        img_ratio = img_width / img_height
        ph_ratio = ph_width / ph_height
        
        # Calculate scaled dimensions to fit within placeholder
        if img_ratio > ph_ratio:
            # Image is wider - fit to width
            new_width = ph_width
            new_height = int(ph_width / img_ratio)
        else:
            # Image is taller - fit to height
            new_height = ph_height
            new_width = int(ph_height * img_ratio)
        
        # Center the image in the placeholder area
        left = ph_left + (ph_width - new_width) // 2
        top = ph_top + (ph_height - new_height) // 2
        
        return left, top, new_width, new_height

def split_slide_if_needed(slide_info, max_bullets=4, max_chars=50):
    """Splits a slide into multiple slides if it has too many bullets or text is too long."""
    bullets = slide_info.get('bullets', [])
    
    # Check if split is needed
    needs_split = len(bullets) > max_bullets
    
    # Also check if any bullet is too long
    if not needs_split:
        for bullet in bullets:
            if len(str(bullet)) > max_chars:
                needs_split = True
                break
    
    if not needs_split:
        return [slide_info]
    
    # Split into multiple slides
    result_slides = []
    chunks = []
    current_chunk = []
    current_length = 0
    
    for bullet in bullets:
        bullet_text = str(bullet)
        # If single bullet is too long, try to keep it alone on a slide
        if len(bullet_text) > max_chars:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
            chunks.append([bullet_text])
        else:
            if len(current_chunk) >= max_bullets:
                chunks.append(current_chunk)
                current_chunk = [bullet_text]
            else:
                current_chunk.append(bullet_text)
    
    if current_chunk:
        chunks.append(current_chunk)
    
    # Create slides from chunks
    base_title = slide_info.get('title', 'Slide')
    total_parts = len(chunks)
    
    for i, chunk in enumerate(chunks):
        new_slide = slide_info.copy()
        if total_parts > 1:
            new_slide['title'] = f"{base_title} ({i+1}/{total_parts})"
        new_slide['bullets'] = chunk
        # Only first slide gets the image
        if i > 0:
            new_slide['image_path'] = None
        result_slides.append(new_slide)
    
    return result_slides

def create_presentation(slides_data, template_path, output_path, md_dir):
    """Constructs the presentation with improved text fitting and image placement."""
    prs = Presentation(template_path)
    
    num_original_slides = len(prs.slides)
    print(f"Template contains {num_original_slides} sample slides. They will be removed after generation.")

    # Pre-process slides to split any that are too large
    processed_slides = []
    for slide_info in slides_data:
        split_slides = split_slide_if_needed(slide_info)
        processed_slides.extend(split_slides)
    
    if len(processed_slides) > len(slides_data):
        print(f"Note: Split {len(slides_data)} slides into {len(processed_slides)} slides for better readability")

    for i, slide_info in enumerate(processed_slides):
        l_idx = slide_info.get('layout_index', 1)
        if l_idx >= len(prs.slide_layouts):
            l_idx = 1
            
        layout = prs.slide_layouts[l_idx]
        slide = prs.slides.add_slide(layout)
        print(f"Creating Slide {i+1}: {slide_info.get('title', 'Untitled')}")

        # Set title with auto-fitting
        if slide.shapes.title:
            slide.shapes.title.text = slide_info.get('title', 'Slide')
            if slide.shapes.title.has_text_frame:
                auto_fit_text(slide.shapes.title.text_frame, max_font_size=32, min_font_size=20)

        # Find and populate text placeholder
        text_placeholder = next(
            (s for s in slide.placeholders if s.placeholder_format.type in 
             [PP_PLACEHOLDER.BODY, PP_PLACEHOLDER.SUBTITLE, PP_PLACEHOLDER.OBJECT]), 
            None
        )
        
        if text_placeholder and slide_info.get('bullets'):
            tf = text_placeholder.text_frame
            tf.clear()
            tf.word_wrap = True
            
            for bullet_text in slide_info.get('bullets', []):
                p = tf.add_paragraph()
                p.text = str(bullet_text)
                p.level = 0
            
            # Auto-fit text content
            auto_fit_text(tf, max_font_size=18, min_font_size=10)

        # Handle image placement
        img_rel = slide_info.get('image_path')
        if img_rel:
            full_img_path = os.path.normpath(os.path.join(md_dir, img_rel))
            if os.path.exists(full_img_path):
                try:
                    # Try to find picture placeholder first
                    pic_placeholder = next(
                        (s for s in slide.placeholders if s.placeholder_format.type == PP_PLACEHOLDER.PICTURE), 
                        None
                    )
                    
                    if pic_placeholder:
                        # Use placeholder with aspect ratio fitting
                        left, top, width, height = fit_image_to_placeholder(pic_placeholder, full_img_path)
                        # Remove placeholder and add picture in its place
                        sp = pic_placeholder._element
                        sp.getparent().remove(sp)
                        slide.shapes.add_picture(full_img_path, left, top, width=width, height=height)
                    else:
                        # No placeholder - add to default position with reasonable size
                        slide.shapes.add_picture(
                            full_img_path, 
                            Inches(6), 
                            Inches(1.5), 
                            width=Inches(3.5)
                        )
                except Exception as e:
                    print(f"Warning: Could not insert image {img_rel}: {e}")

    # Remove original template slides
    print(f"Cleaning up {num_original_slides} sample slides...")
    for _ in range(num_original_slides):
        delete_slide(prs, 0)

    try:
        prs.save(output_path)
        print(f"\n✓ SUCCESS: Created presentation at {output_path}")
    except PermissionError:
        print(f"\n❌ ERROR: Cannot save to '{output_path}'")
        print("The file may have been opened during processing.")
        print("Please close it and run the script again.")
        sys.exit(1)

def validate_files(md_path, pptx_path):
    if not os.path.exists(md_path):
        print(f"Error: Markdown file not found: {md_path}")
        sys.exit(1)
    if not os.path.exists(pptx_path):
        print(f"Error: PowerPoint template not found: {pptx_path}")
        sys.exit(1)

def check_output_writable(output_path):
    """Check if output file can be written (not open in another program)."""
    if os.path.exists(output_path):
        try:
            # Try to open file in append mode to check if it's locked
            with open(output_path, 'a'):
                pass
            return True
        except PermissionError:
            return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Convert Markdown to PowerPoint with AI-powered layout selection")
    parser.add_argument("--md", required=True, help="Path to input Markdown file")
    parser.add_argument("--pptx", required=True, help="Path to PowerPoint template")
    parser.add_argument("--api_key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--output", help="Output file path (default: adds '_output' to template name)")
    args = parser.parse_args()

    validate_files(args.md, args.pptx)
    
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key required. Use --api_key or set OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    md_abs = os.path.abspath(args.md)
    md_dir = os.path.dirname(md_abs)
    
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.pptx)
        output_path = f"{base}_output{ext}"
    
    # Check if output file is writable before doing all the work
    if not check_output_writable(output_path):
        print(f"\n❌ ERROR: Cannot write to '{output_path}'")
        print("The file may be open in PowerPoint or another program.")
        print("Please close the file and try again.\n")
        sys.exit(1)

    with open(md_abs, 'r', encoding='utf-8') as f:
        content = f.read()

    _, images = parse_markdown(content)
    layouts = get_template_layouts(args.pptx)
    
    print("Analyzing content and generating slide structure...")
    slides_json = summarize_and_map_with_llm(content, layouts, images, api_key)
    
    print(f"Creating presentation with {len(slides_json)} slides...")
    create_presentation(slides_json, args.pptx, output_path, md_dir)

if __name__ == "__main__":
    main()