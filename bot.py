import os
from dotenv import load_dotenv
load_dotenv()
token = os.getenv("DISCORD_TOKEN")

import discord
from discord.ext import commands
from discord import Interaction
from doc_insights import get_pdf_text, get_text_chunks, get_vector_store, user_input
import asyncio




# def perform_pdf_task(pdf_docs):
#     raw_text = get_pdf_text(pdf_docs)
#     text_chunks = get_text_chunks(raw_text)
#     get_vector_store(text_chunks)

def get_answer(question):
    ans =user_input(question)
    return ans

from PyPDF2 import PdfReader
def perform_pdf_task(pdf):
    text=""
    pdf_reader= PdfReader(pdf)
    for page in pdf_reader.pages:
        text+= page.extract_text()
    text_chunks = get_text_chunks(text)
    get_vector_store(text_chunks)
# def get_answer(question):
#     # This is where you'd put your code to get the answer to the question
#     return "This is a placeholder answer"


client = commands.Bot(command_prefix="!",intents=discord.Intents.all())

@client.event
async def on_ready():
    await client.tree.sync()
    # await client.change_presence(activity=discord.activity.Game(name="seggs"),status=discord.Status.do_not_disturb)
    await client.change_presence(activity=discord.activity.Game(name="with docs"))
    print(f"{client.user.name} is logged in ")

# @client.command()
# async def hello(ctx):
#     await ctx.send("hey there!")

# @client.tree.command(name="ping",description="it will show ping!")
# async def ping(interaction: Interaction):
#     bot_latency = round(client.latency*1000)
#     await interaction.response.send_message(f"Pong!... {bot_latency}ms")

@client.command()
async def upload(ctx):
    if ctx.message.attachments:
        for file in ctx.message.attachments:
            await file.save(file.filename)
            # Now you have the document saved and you can process it
            # with your PDF reader and GenAI API
            perform_pdf_task(file.filename)
            await ctx.send('Document received. Please provide your question.')
            while True:
                def check(m):
                    return m.author == ctx.author
                question = await client.wait_for('message', check=check)
                if question.content.lower()=='stop':
                    break
                loading_message = await ctx.send("Wait kar thoda bhai ! Patience is the key bro")
                await asyncio.sleep(2)
                answer = get_answer(question.content)
                # await ctx.send(answer)
                await loading_message.edit(content=answer)
client.run(token)