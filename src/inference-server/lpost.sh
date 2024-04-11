
curl -X POST '127.0.0.1:8080/v1/completions' \
     -H 'Content-Type: application/json' \
     -d '{
             "persona_id": "yuki_hinashi_en",
	     "prompt": "How are you doing?",
	     "chat_history": []
     }'

