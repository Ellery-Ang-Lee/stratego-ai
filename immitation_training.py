import stratego_env
import xml.etree.ElementTree as ET
import time

#board needs to be mirrored up down (probably a problem with the environment)

def main():
    tree = ET.parse('data/strados2015-2/classic-2015.2-2341.xml')
    #tree = ET.parse('data/strados2005-4/classic-2005.4-568.xml')
    #tree = ET.parse('data/strados2005-5/classic-2005.5-5771.xml')

    root = tree.getroot()
    
    env = stratego_env.StrategoEnv()
    
    for game in root.findall('game'):
        setup = game.find('field').get("content")[::-1]
        temp = ""
        for i in range(10):
            temp += setup[(i * 10):10 + (i * 10)][::-1]
        setup = temp
        #setup = game.find('field').get("content")
    
        env.reset(red_setup=setup[:60], blue_setup=setup[60:])
        print(env.board)
        env.render(True)
        for move in game.findall('move'):
            env.step(move.get("source") + "-" + move.get("target"))
            env.render(True)
            time.sleep(0.5)


main()