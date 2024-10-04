from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)
CORS(app)

model_name = "facebook/blenderbot-400M-distill"

# Load model (download on first run and reference local installation for consequent runs)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

conversation_history = []

@app.route('/chatbot', methods=['POST'])
def handle_prompt():
    data = request.json
    input_text = data.get('prompt', '')

    # Create conversation history string
    history_string = "\n".join(conversation_history)

    # Tokenize the input text and history
    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")

    # Generate the response from the model
    outputs = model.generate(**inputs)

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Add interaction to conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)