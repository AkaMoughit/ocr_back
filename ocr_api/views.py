from django.shortcuts import render
from rest_framework.decorators import api_view
# Create your views here.
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display
from dotenv import load_dotenv
  # take environment variables from .env.
import google.generativeai as genai
import json
from django.http import JsonResponse
import os
@api_view(['POST'])
def ocr(request):
    load_dotenv()
    wf = Workflow()

    # Add text detection algorithm
    text_det = wf.add_task(name="infer_mmlab_text_detection", auto_connect=True)

    # Add text recognition algorithm
    text_rec = wf.add_task(name="infer_mmlab_text_recognition", auto_connect=True)
    text_rec.set_parameters({
        "model_name": "sar",
        "cfg": "sar_resnet31_parallel-decoder_5e_st-sub_mj-sub_sa_real.py",
    })
    text_det.set_parameters({
        "model_name": "dbnetpp",
        "cfg": "dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015.py",
    })

    img = request.data['imageUrl']
    wf.run_on(url=img)

    # Display results
    img_output = text_rec.get_output(0)
    recognition_output = text_rec.get_output(1)


    output_data = text_rec.get_output(1).to_json(
        # options= ['json_format', 'indented']
    )
    data = json.loads(output_data)

    # Iterate over the 'fields' in the data
    for i, field in enumerate(data['fields']):
        # Create a new dictionary that only contains the 'text' and 'confidence'
        simplified_field = {'text': field['text'], 'confidence': field['confidence']}
        # Replace the original field with the simplified field
        data['fields'][i] = simplified_field
    os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    def get_gemini_response(prompt):
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content([prompt])
        return response.text
    prompt = f"""
    # Example of a Moroccan name: Omar 
    Given the following dictionary:
    {data}
    extract the values that contain a moroccan name  and an ID number in the format (2 letters then 6 numbers) ,
    and return them as name=name_extracted and id=id_extracted
    make sure to return only the the name id don't return any other text:
    """
    response=get_gemini_response(prompt)
    print(response)
    lines = response.split('\n')
    result = {}
    for line in lines:
        parts = line.split('=')
        if len(parts) == 2:
            key = parts[0].strip()
            value = parts[1].strip()
            if key in ['name', 'id']:
                result[key] = value
    return JsonResponse(result)