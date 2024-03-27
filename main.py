from src.agent import SearchAgent
from src.bot import DiscordBot

from dotenv import load_dotenv

load_dotenv()

agent = SearchAgent()

bot = DiscordBot(agent)
bot.run()