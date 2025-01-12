import sys
from pathlib import Path

# from django.conf import settings
# BASE_DIR = Path(__file__).resolve().parent.parent.parent
# MODEL_PATH = Path(BASE_DIR) / "src"
# sys.path.append(str(MODEL_PATH))

from .predict import predict_fn
from .serializers import PredictSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

class PredictAPIView(APIView):

    def post(self, request):
        
        serializer = PredictSerializer(data=request.data)
        
        if serializer.is_valid():
            
            text = serializer.validated_data["text"]
            pred = predict_fn(text)
            
            return Response({"text": text, "output": pred}, status=status.HTTP_200_OK)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class BatchPredictView(APIView):
    pass

class ModelInfoView(APIView):
    pass

class ExplainPredictView(APIView):
    pass


'''
# Model Serving points
# (1) POST: Predict -> Single endpont
# (2) POST: Batch predictions -> File could be uploaded .i.e. json, csv, excel containing texts (with a single column named `text`)
# (3) GET: Model-Info -> The date the model was trained, model version number, metrics and its name
# (4) POST: explain-predictions -> Explain single text prediction be it LIME or SHAP values
# (5) GET: List all available models and rank them based on average inference scores
# (6) POST: User feedback -> Explaination in words how the choosen model predicts based on input text.
# (7) GET: Documentation -> Retrieve documentation explaining how the system operates.
'''
