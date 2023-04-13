import os
import warnings
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText
from transformers import MvpTokenizer, MvpForConditionalGeneration
from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time


############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
    pass


tokenizer = MvpTokenizer.from_pretrained("RUCAIBox/mvp")
model = MvpForConditionalGeneration.from_pretrained("RUCAIBox/mtl-question-answering")


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    output = []
    for text in request.text:
        inputs = tokenizer(
            text,
            return_tensors="pt", )
        generated_ids = model.generate(**inputs)
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        output.append(response)

    return SimpleText(dict(text=output))
