import discord
from discord.ext import commands   
from nn import Package
import asyncio
import math
from bs4 import BeautifulSoup
import aiohttp

def truncate(f, n):
    return math.floor(f * 10 ** n) / 10 ** n

class NNCog(commands.Cog):
    def __init__(self,bot):
        self.bot = bot
        self.model_name = None
        self._load()
        self.bar_end = [
            (
                '<:bar_1_01:791340155944304670>',
                '<:bar_2_01:791340155973533696>'
            ),
            (
                '<:bar_3_02:791340155889647637>',
                '<:bar_4_02:791340155620818945>'
            ),
            (
                '<:bar_5_03:791340155981660160>',
                '<:bar_6_03:791340156027797504>'
            ),
            (
                '<:bar_7_04:791340156053225534>',
                '<:bar_8_04:791340155768012831>'
            ),
            (
                '<:bar_9_05:791340155982184468>',
                '<:bar_10_05:791340155989393418>'
            ),
            (
                '<:bar_11_06:791340155776532541>',
                '<:bar_12_06:791340156036186172>'
            ),
            (
                '<:bar_13_07:791340156035924008>',
                '<:bar_14_07:791340156371337216>'
            ),
            (
                '<:bar_15_08:791340156010364998>',
                '<:bar_16_08:791340155632877579>'
            ),
            (
                '<:bar_17_09:791340156174598154>',
                '<:bar_18_09:791340155931066389>'
            ),
            (
                '<:bar_19_10:791340155997782017>',
                '<:bar_20_10:791340156158345287>'
            )
        ]
        
        self.bar_full = [
            '<:bar_full_01:791340156099100682>',
            '<:bar_full_02:791340156136325160>',
            '<:bar_full_03:791340155696840705>',
            '<:bar_full_04:791340156287320145>',
            '<:bar_full_05:791340155758706699>',
            '<:bar_full_06:791340155784134738>',
            '<:bar_full_07:791340156291776532>',
            '<:bar_full_08:791340155943387137>',
            '<:bar_full_09:791340155926741013>',
            '<:bar_full_10:791340155990441995>'
        ]
        self.bar_void = [
            '<:bar_void_01:791340156190720020>',
            '<:bar_void_02:791340156317728778>',
            '<:bar_void_03:791340156565192714>',
            '<:bar_void_04:791340156166471680>',
            '<:bar_void_05:791340155792392223>',
            '<:bar_void_06:791340155923333134>',
            '<:bar_void_07:791340156316680202>',
            '<:bar_void_08:791340156229517312>',
            '<:bar_void_09:791340156199895041>',
            '<:bar_void_10:791340156262547496>'
        ]
        self.hand = '<:hand:791142229518712893>'
        self.blank = '<:blank:551400844654936095>'
        
    def _load(self):
        self.package = Package.load(self.model_name)
        
    @commands.command()
    async def load(self, ctx, name=None):
        self.model_name = name
        self._load()
    
    def get_bar(self, perc):
        prc = int(round(perc*100, 1))
        start_idx = int(10*perc)
        fbl = start_idx
        end_idx2 = int(round((10*perc - start_idx)))
                
        full_bar = ''.join(self.bar_full[:fbl])
        end_bar = self.bar_end[fbl][end_idx2] if prc >= 1 and not fbl == 10 else ''
        void_bar = ''.join(self.bar_void[start_idx+(1 if prc else 0):]) if fbl < 10 else ''
        
        bar_txt = full_bar + end_bar + void_bar
        return bar_txt
    
    def get_response(self, text, message=None, quote=False):
        pred = self.package.predict(text)[0][0]
        if message:
            print(f'[{message.author.display_name}] [{pred}]: {text}')
        else:
            print(f'[UNKNOWN] [{pred}]: {text}')
        pred = round(pred, 3)
        
        q = f'`{text}`\n' if quote else ''
        t = f'{q}{self.hand}  **CLICKBAIT METER**  >  {self.get_bar(pred)}  |  **{round(pred*100,1)}%**'
        return t
    
    @commands.command()
    async def bar(self, ctx, pred:float):
        await ctx.send(self.get_bar(pred))
    
    async def title(self, url):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                title = soup.find("meta",  property="og:title")
                title = title.get('content') if title else None
                if not title:
                    t = soup.find('title')
                    if t:
                        title = t.string
                return title or url
      

    def is_url(self, text):
        return text.startswith('http://') or text.startswith('https://')
                
    @commands.command(aliases=['predict','pred','cb','clickbait','bait'])
    async def c(self, ctx, *, text):
        message = ctx.message
        if self.is_url(message.content):
            await message.channel.send(content=self.get_response(await self.title(message.content), message, quote=True), reference=message)
        else:
            await message.channel.send(content=self.get_response(message.content, message), reference=message)

    
    @commands.Cog.listener()
    async def on_message(self, message):
        if message.author.bot:
            return
        if self.is_url(message.content):
            await message.channel.send(content=self.get_response(await self.title(message.content), message, quote=True), reference=message)
        elif isinstance(message.channel, discord.abc.PrivateChannel):
            await message.channel.send(content=self.get_response(message.content, message), reference=message)
            
            
def setup(bot):
    bot.add_cog(NNCog(bot))