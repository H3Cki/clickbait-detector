import os
import sys
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import discord
from discord.ext import commands
import sys, traceback
from datetime import datetime
import platform


def start_bot():

    initial_extensions = ['cogs.nncog']
    bot = commands.Bot(command_prefix="!", description=f'Nazywam się Arkowiec, jestem najlepszym botem na świecie.\n\nOS: {platform.system()}\nPlatform: {platform.platform()}\nVersion: {platform.version()}\nCPU: {platform.processor()}')


    if __name__ == '__main__':
        for extension in initial_extensions:
            try:
                bot.load_extension(extension)
            except Exception as e:
                print(f'Failed to load extension {extension}.', file=sys.stderr)
                traceback.print_exc()

    @bot.command()
    async def unloadex(ctx,cog):
        try:
            bot.unload_extension(cog)
            await ctx.send(embed=discord.Embed(description=f'Cog {cog} unloaded successfully',color=discord.Color.from_rgb(26,230,80)))
        except Exception as e:
            await ctx.send(embed=discord.Embed(description=f'{cog} could not be unloaded:\n\n{str(e)}',color=discord.Color.from_rgb(230,10,80)))
            
    @bot.command()
    async def loadex(ctx,cog):
        try:
            bot.load_extension(cog)
            await ctx.send(embed=discord.Embed(description=f'Cog {cog} loaded successfully',color=discord.Color.from_rgb(26,230,80)))
        except Exception as e:
            await ctx.send(embed=discord.Embed(description=f'{cog} could not be loaded:\n\n{str(e)}',color=discord.Color.from_rgb(230,10,80)))
            
    @bot.command()
    async def reloadex(ctx,cog):
        try:
            bot.unload_extension(cog)
            await ctx.send(embed=discord.Embed(description=f'Cog {cog} unloaded successfully',color=discord.Color.from_rgb(26,230,80)))
        except Exception as e:
            await ctx.send(embed=discord.Embed(description=f'{cog} could not be unloaded:\n\n{str(e)}',color=discord.Color.from_rgb(230,10,80)))
            
        try:
            bot.load_extension(cog)
            await ctx.send(embed=discord.Embed(description=f'Cog {cog} loaded successfully',color=discord.Color.from_rgb(26,230,80)))
        except Exception as e:
            await ctx.send(embed=discord.Embed(description=f'{cog} could not be loaded:\n\n{str(e)}',color=discord.Color.from_rgb(230,10,80)))

    @bot.event
    async def on_ready():
        """http://discordpy.readthedocs.io/en/rewrite/api.html#discord.on_ready"""
        print(f'\n\nLogged in as: {bot.user.name} - {bot.user.id}\nVersion: {discord.__version__}\n')
        print(f'[{datetime.now()}] Successfully logged in and booted...!')


    bot.run("NjYzMTk5MzQ1OTE0NTQ0MTM4.XhFChQ.NGp9WTCP937Di9obLsN6D3tzn2M",reconnect=True)

if __name__ == '__main__':
    start_bot()