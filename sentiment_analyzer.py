import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentAnalyzer:
    def __init__(self, model_name="yiyanghkust/finbert-tone"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Sentiment Model ({model_name}) on {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.labels = ['Neutral', 'Positive', 'Negative']
            print("Sentiment Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def analyze(self, text):
        """
        Analyzes text by splitting it into chunks.
        Strategy: 'Strongest Signal' (Max-Pooling). 
        The chunk with the highest Positive or Negative confidence dictates the result.
        """
        if not self.model or not text.strip():
            return 0.0, "Neutral"

        # okenize & Split into 510-token chunks
        tokens = self.tokenizer(text, add_special_tokens=False, return_tensors='pt')
        input_id_chunks = tokens['input_ids'][0].split(510)
        mask_chunks = tokens['attention_mask'][0].split(510)

        best_sentiment = "Neutral"
        best_score = -1.0

        # Analyze each chunk individually
        for i in range(len(input_id_chunks)):
            # Re-add special tokens [CLS] and [SEP]
            input_ids = torch.cat([
                torch.tensor([101]).to(self.device), 
                input_id_chunks[i].to(self.device),
                torch.tensor([102]).to(self.device)
            ]).unsqueeze(0)

            attention_mask = torch.cat([
                torch.tensor([1]).to(self.device),
                mask_chunks[i].to(self.device),
                torch.tensor([1]).to(self.device)
            ]).unsqueeze(0)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Get probabilities: [Neutral, Positive, Negative]
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            
            # Extract scores
            neutral_score = probs[0]
            positive_score = probs[1]
            negative_score = probs[2]
            
            # --- THE SMART LOGIC ---
            
            # Check if this chunk is STRONGLY Positive
            if positive_score > best_score and positive_score > neutral_score:
                best_score = positive_score
                best_sentiment = "Positive"

            # Check if this chunk is STRONGLY Negative
            if negative_score > best_score and negative_score > neutral_score:
                best_score = negative_score
                best_sentiment = "Negative"

            # (We intentionally ignore Neutral scores unless they are the only thing we found)

        # Fallback: If no strong positive/negative signal was found, return Neutral
        if best_sentiment == "Neutral":
             return 0.0, "Neutral"
             
        return best_score, best_sentiment
    
