import numpy as np

# Feature engineering
def make_np_float(feature):
    return np.array(feature).astype(np.float32)

def featurize_map(obs):
    """
    feature_size_map = {
        'board': len(board),
        'bomb_blast_strength': len(bomb_blast_strength),
        'bomb_life': len(bomb_life),
        'position': len(position),
        'ammo': len(ammo),
        'blast_strength': len(blast_strength),
        'can_kick': len(can_kick),
        'teammate': len(teammate),
        'enemies': len(enemies)
    }
    feature_map = {
        'board': board,
        'bomb_blast_strength': bomb_blast_strength,
        'bomb_life': bomb_life,
        'position': position,
        'ammo': ammo,
        'blast_strength': blast_strength,
        'can_kick': can_kick,
        'teammate': teammate,
        'enemies': enemies
    }
    """
    board = obs["board"].reshape(11,11).astype(np.float32)
    bomb_blast_strength = obs["bomb_blast_strength"].reshape(11,11).astype(np.float32)
    bomb_life = obs["bomb_life"].reshape(11,11).astype(np.float32)
    position = make_np_float(obs["position"])
    ammo = make_np_float([obs["ammo"]])
    blast_strength = make_np_float([obs["blast_strength"]])
    can_kick = make_np_float([obs["can_kick"]])

    teammate = obs["teammate"]
    if teammate is not None:
        teammate = teammate.value
    else:
        teammate = -1
    teammate = make_np_float([teammate])

    enemies = obs["enemies"]
    enemies = [e.value for e in enemies]
    if len(enemies) < 3:
        enemies = enemies + [-1]*(3 - len(enemies))
    enemies = make_np_float(enemies)
    
    feature_map = {
        'board': board,
        'bomb_blast_strength': bomb_blast_strength,
        'bomb_life': bomb_life,
        'position': position,
        'ammo': ammo,
        'blast_strength': blast_strength,
        'can_kick': can_kick,
        'teammate': teammate,
        'enemies': enemies
    }

    # Some useful rules:
    '''
    from: http://www.cs.huji.ac.il/~ai/projects/2017/learning/pommerman/files/report.pdf
    Valid directions - for each of the 4 directions, True
    means that direction is free and on the board and
    False otherwise.
    • Dangerous bombs - for each direction, if there is no
    bomb in that direction, the value is 0. Otherwise,
    the value is an int representing the time left for the
    bomb to explode.
    • Adjacent flames - for each direction, if there are
    no flames in that direction, the value is False,
    otherwise, True.
    • Current quarter - representing which of the 4 quarters
    of the board the agent is in.
    • Enemy in range of our bomb - True or False
    • Wood wall in range of our bomb - True or False
    • Stands on bomb - True or False
    • Powerups in range of two steps for each direction
    
    from: https://nihal111.github.io/docs/pommerman_report.pdf
    
    
    from: http://www.akhalifa.com/documents/hybrid-search-pommerman.pdf
    This one divides the search problem into different ways.
    
    1. Number of Enemies within range 4, 8, 16
    2. Number of valid directions (avoid getting stuck in the corner)
    3. Enemy in range of our bomb - True or False
    4. Wood wall in range of our bomb - True or False
    5. stands on bomb
    6. range to powerups
    7. range to can kick
    8. bomb count
    9. EvadeCondition: 
    '''
    
    #1. number of enemies within range 4, 8, 16, 20.
    def getManhattanDistance(pos1, pos2):
        # Calculate the Manhattan Distance
        
        
    # Get current position
    for i in range(11):
        for j in range(11):
            board
    #2. closest distance to enemies and its invert.
    
    return feature_map