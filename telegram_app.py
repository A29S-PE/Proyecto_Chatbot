from telegram.ext import Application, MessageHandler, filters
import requests

TOKEN = '8437215326:AAGLKoJjRB_JZ5KNmv46B8Kf_sNYmS-G8mU'
API_URL = 'https://tuapi.com/mensaje'

async def handle_message(update, context):
    user_id = str(update.effective_user.id)
    message = update.message.text
    data = {'user_id': user_id, 'message': message}
    # requests.post(API_URL, json=data)
    await update.message.reply_text(f'Hello {update.effective_user.first_name}')

app = Application.builder().token(TOKEN).build()
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
app.run_polling()