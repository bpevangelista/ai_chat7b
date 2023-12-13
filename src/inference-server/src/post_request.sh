curl -X POST http://localhost:8080/predict \
   -H "Content-Type: application/json" \
   -d '{"message": "Hello, I am Bruno. What are you doing here?", "chat_history": "", "persona_id":"yuki_hinashi_en"}' \
