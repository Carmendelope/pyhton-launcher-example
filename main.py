import numpy as np

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from flask import Flask

output = "<p> Before processing value </p>"
app = Flask(__name__)
@app.route('/')
def serve():
    return output
    
app.run(host='0.0.0.0', port=5000)

print("tokenizers example")
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.pre_tokenizer = Whitespace()

print("Initilize files")
files = [f"./wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]

print("Trainning")
tokenizer.train(files, trainer)
print("save the results")
tokenizer.save("./tokenizer-wiki.json")
output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
print(output.tokens)

print("numpy example")
arr = np.array([1, 2, 3, 4, 5])

print(arr)

print("Example finished!")

output = "<p> Post processing value </p>"
