import stratego_env
import xml.etree.ElementTree as ET

def main():
    tree = ET.parse('data/stratego_games.xml')
    root = tree.getroot()
    
    env = stratego_env.StrategoEnv()
    
    for game in root.findall('game'):
        env.reset()


main()