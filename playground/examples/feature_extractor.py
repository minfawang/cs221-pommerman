import numpy as np
MAX_EPISODE_TIMTESTAMPS = 2000

# Feature engineering
def make_np_float(feature):
    return np.array(feature).astype(np.float32)


def categorical_feature_map(obs):
    """
    Produce 19 categorical feature map.
    
    Learned from https://arxiv.org/pdf/1807.06919.pdf
    
    The first map contains the integer values of each bomb’s blast strength at the location of that bomb.
    The second map is similar but the integer value is the bomb’s remaining life. At all other positions,
    the first two maps are zero. 
    The next map is binary and contains a single one at the agent’s location. If the agent is dead, 
    this map is zero everywhere. 
    The following two maps are similar. One is full
    with the agent’s integer current bomb count, the other with its blast radius. We then have a full binary
    map that is one if the agent can kick and zero otherwise.
    
    The next maps deal with the other agents. The first contains only ones if the agent has a teammate
    and zeros otherwise. This is useful for building agents that can play both team and solo matches. If
    the agent has a teammate, the next map is binary with a one at the teammate’s location (and zero if
    she is not alive). 
    
    Otherwise, the agent has three enemies, so the next map contains the position of the
    enemy that started in the diagonally opposed corner from the agent. The following two maps contain
    the positions of the other two enemies, which are present in both solo and team games.
    
    We then include eight feature maps representing the respective locations of passages, rigid walls,
    wooden walls, flames, extra-bomb power-ups, increase-blast-strength power-ups, and kicking-ability
    power-ups. All are binary with ones at the corresponding locations.
    
    Finally, we include a full map with the float ratio of the current step to the total number of steps. This
    information is useful for distinguishing among observation states that are seemingly very similar,
    but in reality are very different because the game has a fixed ending step where the agent receives
    negative reward for not winning.
    """
    board = obs["board"].reshape(11,11,1).astype(np.float32)
    bomb_blast_strength = obs["bomb_blast_strength"].reshape(11,11,1).astype(np.float32)
    bomb_life = obs["bomb_life"].reshape(11,11,1).astype(np.float32)
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
    
    fmaps = {}
    fmaps['fmap1'] = bomb_blast_strength
    fmaps['fmap2'] = bomb_life
    fmaps['fmap3'] = np.zeros((11, 11, 1))
    fmaps['fmap3'][int(position[0])][int(position[1])] = 1.0
    
    fmaps['fmap4'] = np.ones((11, 11, 1)) * ammo
    fmaps['fmap5'] = np.ones((11, 11, 1)) * blast_strength
    fmaps['fmap6'] = np.ones((11, 11, 1)) * can_kick
    
    fmaps['fmap7'] = (board == enemies[0]).astype(np.float32)
    fmaps['fmap8'] = (board == enemies[1]).astype(np.float32)
    fmaps['fmap9'] = (board == enemies[2]).astype(np.float32)
    
    fmaps['fmap10'] = (board == 0).astype(np.float32) # passage
    fmaps['fmap11'] = (board == 1).astype(np.float32) # rigid walls
    fmaps['fmap12'] = (board == 2).astype(np.float32) # wooden walls
    fmaps['fmap13'] = (board == 4).astype(np.float32) # flames
    fmaps['fmap14'] = (board == 6).astype(np.float32) # extra-bomb power-ups, 
    fmaps['fmap15'] = (board == 7).astype(np.float32) # increase-blast-strength power-ups
    fmaps['fmap16'] = (board == 8).astype(np.float32) # kicking-ability power-ups
    
    fmaps['fmap17'] = np.ones((11, 11, 1)) * float(obs['step_count']) / MAX_EPISODE_TIMTESTAMPS
    return fmaps

def featurize_map(obs):
    """
    A simple rule based feature extractor.
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
    
    def getManhattanDistance(pos1, pos2):
        # Calculate the Manhattan Distance
        dist = float(abs(pos1[1] - pos2[1]) + abs(pos1[0] - pos2[0]))
        return [dist, 2.0/(1.0+dist)]
        
    def isWithinBombRange(bomb_pos, player_pos, strength):
        i, j = bomb_pos
        return ((i==player_pos[0]) and abs(j-player_pos[1]) <= strength) or (
            (j==player_pos[1]) and abs(i-player_pos[0]) <= strength)
    
    enemy_id_map = {
        enemies[id]: id
        for id in range(len(enemies))
    }
    
    # 1. distance to enemies
    dist_to_enemies = [0.0] * 6
    # 2. number of enemies within range 4, 8, 16, 20.
    enemies_within_range = [0] * 4
    # 3. bomb info, for each of the following, we keep 10 only
    bomb_have_player_in_range = [] #[0] * 4
    bomb_have_wood_in_range = [] #0
    bomb_strength = [] #0
    bomb_lives = [] #0
    # 4. number of valid directions
    num_valid_move = 0 #0
    # 5. distance to closest super power
    dist_to_ammo = [-1, -1]
    dist_to_kick = [-1, -1]
    dist_to_strength = [-1, -1]
    
    for i in range(11):
        for j in range(11):
            if board[i][j] in enemies:
                # found an enemy
                real_dist, invert_dist = getManhattanDistance(position, (i, j))
                dist_to_enemies[enemy_id_map[board[i][j]]] = real_dist
                dist_to_enemies[enemy_id_map[board[i][j]]*2+1] = invert_dist
                
                if real_dist < 4:
                    enemies_within_range[0] += 1
                if real_dist < 8:
                    enemies_within_range[1] += 1
                if real_dist < 16:
                    enemies_within_range[2] += 1
                if real_dist < 24:
                    enemies_within_range[3] += 1
            
            if board[i][j] == 3 and len(bomb_lives) < 10:
                # found a bomb
                player_in_range = [0.0] * 4
                wood_in_range = 0.0
                
                for dist in range(int(-bomb_blast_strength[i][j]), int(bomb_blast_strength[i][j])+1):
                    new_i = max(min(0, dist + i), 10)
                    new_j = max(min(0, dist + j), 10)
                    if board[new_i][j] in enemies:
                        player_in_range[enemy_id_map[board[new_i][j]]]=1.0
                    if board[i][new_j] in enemies:
                        player_in_range[enemy_id_map[board[i][new_j]]]=1.0
                    if board[new_i][j] == 2 or board[i][new_j] == 2:
                        wood_in_range = 1.0
                        
                bomb_have_player_in_range.extend(player_in_range)
                bomb_have_wood_in_range.append(wood_in_range)
                bomb_strength.append(bomb_blast_strength[i][j])
                bomb_lives.append(bomb_life[i][j])
                
            if i==position[0] and j==position[1]:
                for dist in [-1, 1]:
                    new_i = max(min(0, dist + i), 10)
                    new_j = max(min(0, dist + j), 10)
                    if board[new_i][j] == 0:
                        num_valid_move += 1
                    if board[i][new_j] == 0:
                        num_valid_move += 1
            
            if board[i][j] == 6:
                real_dist, invert_dist = getManhattanDistance((i, j), position)
                if invert_dist >= dist_to_ammo[1]:
                    dist_to_ammo = [real_dist, invert_dist]
            if board[i][j] == 7:
                real_dist, invert_dist = getManhattanDistance((i, j), position)
                if invert_dist >= dist_to_kick[1]:
                    dist_to_kick = [real_dist, invert_dist]
            if board[i][j] == 8:
                real_dist, invert_dist = getManhattanDistance((i, j), position)
                if invert_dist >= dist_to_strength[1]:
                    dist_to_strength = [real_dist, invert_dist]
                    
    # Do some padding of the bomb data, make them of fixed length 10
    for _ in range(10 - len(bomb_lives)):
        bomb_have_player_in_range.extend([0.0] * 4)
        bomb_have_wood_in_range.append(0.0)
        bomb_strength.append(0.0)
        bomb_lives.append(0.0)
    
    # Done.
    feature_map = {
        'dist_to_enemies': make_np_float(dist_to_enemies),
        'enemies_within_range': make_np_float(enemies_within_range),
        'bomb_have_player_in_range': make_np_float(bomb_have_player_in_range),
        'bomb_have_wood_in_range': make_np_float(bomb_have_wood_in_range),
        'bomb_lives': make_np_float(bomb_lives),
        'num_valid_move': make_np_float([num_valid_move]),
        'dist_to_ammo': make_np_float(dist_to_ammo),
        'dist_to_kick': make_np_float(dist_to_kick),
        'dist_to_strength': make_np_float(dist_to_strength),
        'ammo': ammo,
        'blast_strength': blast_strength,
        'can_kick': can_kick
    }
    return feature_map