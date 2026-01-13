import os
import re
import json
import argparse
import sys
from pptx import Presentation
from pptx.util import Inches
from pptx.enum.shapes import PP_PLACEHOLDER
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
    """Extracts metadata from the template to help the LLM select layouts."""
    prs = Presentation(pptx_path)
    layouts = []
    for i, layout in enumerate(prs.slide_layouts):
        placeholders = [p.name for p in layout.placeholders]
        layouts.append({
            "index": i,
            "name": layout.name,
            "placeholders": placeholders
        })
    return layouts

def summarize_and_map_with_llm(md_content, layouts, image_list, api_key):
    """Consults the LLM to structure the report and pick the best template layouts."""
    client = OpenAI(api_key=api_key)
    layout_context = "\n".join([f"- Index {l['index']}: '{l['name']}' (Has: {', '.join(l['placeholders'])})" for l in layouts])
    
    prompt = f"""
    You are a professional presentation designer. Convert the following report into a slide deck.
    
    AVAILABLE TEMPLATE LAYOUTS:
    {layout_context}
    
    AVAILABLE IMAGES:
    {", ".join(image_list)}
    
    INSTRUCTIONS:
    1. Summarize content into concise, professional bullet points.
    2. For each slide, return a JSON object with: "title", "bullets" (list), "image_path" (path or null), and "layout_index".
    3. Return ONLY a JSON object with the key 'slides'.
    
    REPORT:
    {md_content}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a PowerPoint automation expert. Output valid JSON only."},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content).get('slides', [])

def create_presentation(slides_data, template_path, output_path, md_dir):
    """Constructs the presentation and PURGES original template slides."""
    prs = Presentation(template_path)
    
    # 1. Store how many slides were originally in the template
    num_original_slides = len(prs.slides)
    print(f"Template contains {num_original_slides} sample slides. They will be removed after generation.")

    # 2. Add the NEW slides (these will be appended at the end)
    for i, slide_info in enumerate(slides_data):
        l_idx = slide_info.get('layout_index', 1)
        if l_idx >= len(prs.slide_layouts): l_idx = 1
            
        layout = prs.slide_layouts[l_idx]
        slide = prs.slides.add_slide(layout)
        print(f"Creating New Slide {i+1}: {slide_info.get('title')}")

        if slide.shapes.title:
            slide.shapes.title.text = slide_info.get('title', 'Slide')

        text_placeholder = next((s for s in slide.placeholders if s.placeholder_format.type in 
                                [PP_PLACEHOLDER.BODY, PP_PLACEHOLDER.SUBTITLE, PP_PLACEHOLDER.OBJECT]), None)
        
        if text_placeholder:
            tf = text_placeholder.text_frame
            tf.clear()
            for bullet in slide_info.get('bullets', []):
                p = tf.add_paragraph()
                p.text = str(bullet)

        img_rel = slide_info.get('image_path')
        if img_rel:
            full_img_path = os.path.normpath(os.path.join(md_dir, img_rel))
            if os.path.exists(full_img_path):
                pic_placeholder = next((s for s in slide.placeholders if s.placeholder_format.type == PP_PLACEHOLDER.PICTURE), None)
                if pic_placeholder:
                    pic_placeholder.insert_picture(full_img_path)
                else:
                    slide.shapes.add_picture(full_img_path, Inches(6.1), Inches(1.5), width=Inches(3.6))

    # 3. PURGE Logic: Remove the original sample slides
    # We delete from index 0, 'num_original_slides' times.
    print(f"Cleaning up {num_original_slides} sample slides...")
    for _ in range(num_original_slides):
        delete_slide(prs, 0)

    prs.save(output_path)
    print(f"\nSUCCESS: Created clean presentation at {output_path}")

def validate_files(md_path, pptx_path):
    if not os.path.exists(md_path) or not os.path.exists(pptx_path):
        print("Error: Files not found. Check your paths.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--md", required=True)
    parser.add_argument("--pptx", required=True)
    parser.add_argument("--api_key")
    args = parser.parse_args()

    validate_files(args.md, args.pptx)
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    md_abs = os.path.abspath(args.md)
    md_dir = os.path.dirname(md_abs)
    base, ext = os.path.splitext(args.pptx)
    output_path = f"{base}_output{ext}"

    with open(md_abs, 'r', encoding='utf-8') as f:
        content = f.read()

    _, images = parse_markdown(content)
    layouts = get_template_layouts(args.pptx)
    slides_json = summarize_and_map_with_llm(content, layouts, images, api_key)
    create_presentation(slides_json, args.pptx, output_path, md_dir)

if __name__ == "__main__":
    main()