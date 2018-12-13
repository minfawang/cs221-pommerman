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
    9. EvadeCondition
    10. AttackCondition
    11. ExplorationCondition
    '''
    def getDistance(pos1, pos2):
        '''Calculate the Manhattan distance between two position.

        A better solution might be to run BFS to get the real distance between two points, since 
        some cells are blocked.
        '''
        mh_dist = float(abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1]))
        return mh_dist, (1.0)/(mh_dist+1.0)
    
    def withinBombRange(bomb_pos, player_pos, strength):
        i, j = bomb_pos
        return (abs(i-player_pos[0]) <= strength and j == player_pos[1]) or (
            abs(j-player_pos[1]) <= strength and i == player_pos[0])
    
    dist_to_enemies=[-1.0] * 6
    num_enemies_within_range=[0.0] * 3
    valid_direction=0.0
    num_wood = 0.0
    
    dist_to_bomb=[] # distance to 10 bombs
    live_of_bomb=[] # live of 10 bombs
    strength_of_bomb=[]
    within_bomb_range=[]
    wood_wall_in_range_of_bomb=0.0
    enemy_in_range_of_bomb=0.0

    dist_to_powerup=0.0
    dist_to_kick=0.0


    # Get current position
    # https://github.com/MultiAgentLearning/playground/blob/master/docs/environment.md
    enemy_id_map = {}
    for e in range(len(enemies)):
        enemy_id_map[enemies[e]]=e

    for i in range(11):
        for j in range(11):
            if getDistance(position, (i, j))[0] == 1.0:
                if board[i][j] == 0:
                    valid_direction += 1.0

            if board[i][j] in enemies:
                '''Instead of keeping status for each player, for simplicity, we only 
                keep track of the status for our own agent.'''
                enemy = getDistance(position, (i, j))
                dist_to_enemies[enemy_id_map[board[i][j]]] = (enemy[0])
                dist_to_enemies[enemy_id_map[board[i][j]]*2 + 1] = (enemy[1])
                
                if enemy[0] <=4:
                    num_enemies_within_range[0] += 1
                elif enemy[0] <=8:
                    num_enemies_within_range[1] += 1
                elif enemy[0] <=16:
                    num_enemies_within_range[2] += 1
                
            if board[i][j] == 3:
                '''We keep a list of status for each bomb here.'''
                bomb = getDistance(position, (i, j))
                dist_to_bomb.append(bomb[0])
                dist_to_bomb.append(bomb[1])
                live_of_bomb.append(bomb_life[i][j])
                strength_of_bomb.append(bomb_blast_strength[i][j])
                within_bomb_range.append(withinBombRange((i, j), position, bomb_blast_strength[i][j]))
                
                # We also want to know the bomb distance to other players, so our agent
                # can learn how to put itself in a status that is in favor of other players.
            
            if board[i][j] == 2:
                # wood
                num_wood += 1
                
                
            if board[i][j] in [6, 7, 8]:
                # super power
                
    feature_map = {
        
    }
    return feature_map