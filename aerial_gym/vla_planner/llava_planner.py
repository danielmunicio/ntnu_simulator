import ollama
from aerial_gym.vla_planner.base_vla_planner import BaseVLAPlanner
import numpy as np
from PIL import Image
import io
import base64

class LlavaPlanner(BaseVLAPlanner):
    def __init__(self, prompt, history_window=5):
        super().__init__(prompt)
        self.history_window = history_window
        self.message_history = []

    def get_direction(self, image: np.ndarray):
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image.astype('uint8'))

        # Convert to bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()

        # Encode as base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # Add current user message with image to history
        self.message_history.append({
            'role': 'user',
            'content': self.prompt,
            'images': [image_base64]
        })

        # Keep only the last N messages (sliding window)
        # Maintain pairs: keep user messages and their corresponding assistant responses
        if len(self.message_history) > self.history_window * 2:
            self.message_history = self.message_history[-(self.history_window * 2):]

        # Get response from ollama with full conversation history
        response = ollama.chat(model='llava', messages=self.message_history)

        print('--------------------------------------')
        print('LLM Model Response: ')
        print(response['message']['content'])
        print('--------------------------------------')
        # print("Message History: ", self.message_history)
        # Add assistant response to history
        self.message_history.append({
            'role': 'assistant',
            'content': response['message']['content']
        })

        return response['message']['content'].strip()