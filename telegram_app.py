from telegram.ext import Application, MessageHandler, filters
import requests

TOKEN = '8437215326:AAGLKoJjRB_JZ5KNmv46B8Kf_sNYmS-G8mU'
API_URL = 'https://e7d87dd325ea.ngrok-free.app/chat'

async def handle_message(update, context):
    user_id = str(update.effective_user.id)
    message = update.message.text
    data = {'user_id': user_id, 'message': message}
    print(data)
    headers = {
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(API_URL, json=data, headers=headers).json()['response']
    except:
        response = 'Ocurrió un error, por favor vuelve a escribirme 🥺'
    print(response)
    await update.message.reply_text(response)

app = Application.builder().token(TOKEN).build()
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
app.run_polling()