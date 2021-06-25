import random

from django.http import HttpResponse
from django.shortcuts import render
from django.template.loader import render_to_string
from .model.arithmetic_checker import run, run2



def index(request):
    if request.method == "GET":
        return render(request, "index.html")
    else:

        print("post success")


        file_img = request.FILES.get('image')
        ty = request.POST.get('ty')
        data = "statics/" + str(file_img)
        data2 = "static/" + str(file_img)
        with open(data, 'wb') as f:
            for chunk in file_img.chunks():
                f.write(chunk)
        if ty=="1":
            detect_model = "dj/model/detect/output/res50/voc_2007_trainval/arith_anchor_cluster/res50_faster_rcnn_iter_50000.ckpt"
            recog_model = "dj/model/recog/model/arith_aug_128/shadownet_2019-12-12-10-43-39.ckpt-130000"
            char_dict, ord_map_dict = "dj/model/recog/data/char_dict/char_dict.json", "dj/model/recog/data/char_dict/ord_map.json"
            results = run(data, detect_model, recog_model, char_dict, ord_map_dict)
            results = results["results"]
            formulas = [{
                "id": 0,
                "confidence": result['score'],
                "x": result['box'][0],
                "y": result['box'][1],
                "width": result['box'][2]-result['box'][0],
                "height": result['box'][3]-result['box'][1],
                "text": result['recog_str'],
                "result": result['check'],
                "real_result": round(eval(result['recog_str'].split("=")[0]), 4)
            } for i, result in enumerate(results) if not result['check']]
            formulas.sort(key=lambda res: (res['y'], res['x']))
            for i, result in enumerate(formulas):
                result["id"] = i+1
            html = render_to_string('formula_details.html', {'formulas': formulas})
            # print(html)
            import json
            # print(file_img)
            ret={
                "data": html,
                "url": data2
            }
            return HttpResponse(json.dumps(ret))
            # return HttpResponse(html)
        else:
            detect_model = "dj/model/detect/output/yolo_50000.ckpt"
            recog_model = "dj/model/recog/model/arith_aug_128/shadownet-130000"
            char_dict, ord_map_dict = "dj/model/recog/data/char_dict/char_dict.json", "dj/model/recog/data/char_dict/ord_map.json"
            results = run2(str(file_img), detect_model, recog_model, char_dict, ord_map_dict)
            print(results["data"])
            sentences = [{
                "id": result["id"],
                "detect_box": result["detect_box"],
                "recog_str": result["recog_str"],
                "check": result["check"]
            } for result in results["data"]]
            print(sentences)
            html = render_to_string('sentence_details.html', {'sentences': sentences})
            import json
            ret={
                "data": html,
                "url": results["url"]
            }
            print(ret["data"])
            return HttpResponse(json.dumps(ret))



# tensorflow_model_server --port=9000 --model_name=1 --model_base_path=/home/leung-0ng/桌面/AI/final/AI/dj/model/tmp
# tensorflow_model_server --port=9001 --model_name=1 --model_base_path=/home/leung-0ng/桌面/AI/final/AI/dj/model/tmp1