from tensorforce.core.networks import Network
import tensorflow as tf


## DEPRECATED ##
class PommerNet(Network):
    def __init__(self, scope='network', summary_labels=None):
        super().__init__(scope, summary_labels)
        
        self.feature_size_map = {
            'board': 121, 
            'bomb_blast_strength': 121, 
            'bomb_life': 121, 
            'position': 2, 
            'ammo': 1, 
            'blast_strength': 1, 
            'can_kick': 1, 
            'teammate': 1, 
            'enemies': 3
        }
    
    def parse_observation(self, ob):
        """
        np.concatenate((
        board, bomb_blast_strength, bomb_life, position, ammo,
        blast_strength, can_kick, teammate, enemies))
        
        feature_size_map = {
            'board': 121, 
            'bomb_blast_strength': 121, 
            'bomb_life': 121, 
            'position': 2, 
            'ammo': 1, 
            'blast_strength': 1, 
            'can_kick': 1, 
            'teammate': 1, 
            'enemies': 3
        }
        
        feature_map={
        'board': array([ 0.,  0.,  2.,  1.,  0.,  0.,  2.,  2.,  2.,  2.,  2.,  0.,  0.,
       10.,  0.,  2.,  2.,  2.,  0.,  0.,  3.,  2.,  2.,  3.,  0.,  1.,
        1.,  2.,  1.,  0.,  1.,  0.,  1.,  1.,  0.,  1.,  0.,  2.,  0.,
        1.,  1.,  0., 13.,  0.,  0.,  2.,  1.,  2.,  0.,  2.,  1.,  1.,
        1.,  2.,  1.,  0.,  2.,  2.,  0.,  2.,  0.,  1.,  1.,  2.,  2.,
        1.,  2.,  2.,  1.,  1.,  1.,  1.,  0.,  2.,  1.,  2.,  0.,  2.,
        0.,  0.,  1.,  1.,  1.,  2.,  0.,  1.,  3.,  1.,  2.,  0.,  1.,
        0.,  1.,  2.,  1.,  1., 12.,  0.,  0.,  2.,  3.,  0.,  0.,  2.,
        2.,  2.,  0.,  0.,  0.,  0.,  2.,  2.,  1., 11.,  1.,  1.,  0.,
        1.,  0.,  0.,  0.], dtype=float32), 
        
        'bomb_blast_strength': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 2., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0.], dtype=float32), 
       
       'bomb_life': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 5., 0., 0., 6., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 7., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 5., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0.], dtype=float32), 
       
       'position': array([3., 9.], dtype=float32),
       
       'ammo': array([0.], dtype=float32), 
       
       'blast_strength': array([2.], dtype=float32), 
       
       'can_kick': array([0.], dtype=float32), 
       
       'teammate': array([9.], dtype=float32), 
       
       'enemies': array([10., 11., 12.], dtype=float32)}
        """
        print('shape of ob:', ob)
        print(list(self.feature_size_map.keys()))
        print(ob, list(self.feature_size_map.values()))
        print(tf.split(ob, list(self.feature_size_map.values()), 1))
        feature_map = {
            k : v
            for k, v in zip(
                list(self.feature_size_map.keys()), 
                tf.split(ob, list(self.feature_size_map.values()), 1))
        }
        
        for k in {'board', 'bomb_blast_strength', 'bomb_life'}:
            feature_map[k] = tf.reshape(feature_map[k], (-1, 11, 11, 1))
        return feature_map
    
    def build_cnn(self, image):
        # CNN
        image = tf.layers.conv2d(inputs=image, filters=64, 
                                 kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        image = tf.layers.max_pooling2d(inputs=image, pool_size=[2, 2], strides=2)
        
        image = tf.layers.conv2d(inputs=image, filters=32, 
                                 kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        image = tf.layers.max_pooling2d(inputs=image, pool_size=[2, 2], strides=2)
        
        image = tf.layers.flatten(image)
        return image
        
    def tf_apply(self, x, internals, update, return_internals=False):
        
        observation = x['state']
        features = self.parse_observation(observation)
        
        features['board'] = self.build_cnn(features['board'])
        features['bomb_blast_strength'] = self.build_cnn(features['bomb_blast_strength'])
        features['bomb_life'] = self.build_cnn(features['bomb_life'])
        
        print(list(features.values()))
        # Combination        
        x = tf.concat(list(features.values()), 1, name='ConcatAllFeatures')
        
        x = tf.layers.dense(x, 64, tf.nn.relu6)
        x = tf.layers.dense(x, 64, tf.nn.relu6)
        
        if return_internals:
            return x, list()
        else:
            return x

PommerNetBase=PommerNet


def BuildPPONetWorkProfile():
    network = list()
    
    # BUILD CNN
    network.append([
        dict(type='input', names='board'),
        dict(type='conv2d', size=64, window=3, stride=1, padding='SAME', activation='relu'),
        dict(type='pool2d', window=2, stride=2),
        dict(type='conv2d', size=32, window=3, stride=1, padding='SAME', activation='relu'),
        dict(type='pool2d', window=2, stride=2),
        dict(type='flatten'),
        dict(type='output', name='board-cnn')
    ])
    
    # BUILD CNN
    network.append([
        dict(type='input', names='bomb_blast_strength'),
        dict(type='conv2d', size=64, window=3, stride=1, padding='SAME', activation='relu'),
        dict(type='pool2d', window=2, stride=2),
        dict(type='conv2d', size=32, window=3, stride=1, padding='SAME', activation='relu'),
        dict(type='pool2d', window=2, stride=2),
        dict(type='flatten'),
        dict(type='output', name='bomb_blast_strength-cnn')
    ])
    
    # BUILD CNN
    network.append([
        dict(type='input', names='bomb_life'),
        dict(type='conv2d', size=64, window=3, stride=1, padding='SAME', activation='relu'),
        dict(type='pool2d', window=2, stride=2),
        dict(type='conv2d', size=32, window=3, stride=1, padding='SAME', activation='relu'),
        dict(type='pool2d', window=2, stride=2),
        dict(type='flatten'),
        dict(type='output', name='bomb_life-cnn')
    ])   
    
    # Final touch
    network.append([
        dict(type='input', names=['board-cnn', 'bomb_blast_strength-cnn', 'bomb_life-cnn', 
                                  'position', 'ammo', 'blast_strength', 'can_kick', 
                                  'teammate', 'enemies'],
             aggregation_type='concat'),
        dict(type='dense', size=64, activation='relu'),
        dict(type='dense', size=64, activation='relu')
    ])
    
    return network

def BuildBaseNetWorkProfile():
    network = list()
    
    # BUILD CNN
    network.append([
        dict(type='input', names='board'),
        dict(type='conv2d', size=32, window=3, stride=1, padding='SAME', activation='relu'),
        dict(type='pool2d', window=2, stride=2),
        dict(type='conv2d', size=16, window=3, stride=1, padding='SAME', activation='relu'),
        dict(type='pool2d', window=2, stride=2),
        dict(type='flatten'),
        dict(type='output', name='board-cnn')
    ])
    
    # BUILD CNN
    network.append([
        dict(type='input', names='bomb_blast_strength'),
        dict(type='conv2d', size=32, window=3, stride=1, padding='SAME', activation='relu'),
        dict(type='pool2d', window=2, stride=2),
        dict(type='conv2d', size=16, window=3, stride=1, padding='SAME', activation='relu'),
        dict(type='pool2d', window=2, stride=2),
        dict(type='flatten'),
        dict(type='output', name='bomb_blast_strength-cnn')
    ])
    
    # BUILD CNN
    network.append([
        dict(type='input', names='bomb_life'),
        dict(type='conv2d', size=32, window=3, stride=1, padding='SAME', activation='relu'),
        dict(type='pool2d', window=2, stride=2),
        dict(type='conv2d', size=16, window=3, stride=1, padding='SAME', activation='relu'),
        dict(type='pool2d', window=2, stride=2),
        dict(type='flatten'),
        dict(type='output', name='bomb_life-cnn')
    ])   
    
    # Final touch
    network.append([
        dict(type='input', names=['board-cnn', 'bomb_blast_strength-cnn', 'bomb_life-cnn', 
                                  'position', 'ammo', 'blast_strength', 'can_kick', 
                                  'teammate', 'enemies'],
             aggregation_type='concat'),
        dict(type='dense', size=64, activation='relu'),
        dict(type='dense', size=64, activation='relu')
    ])
    return network

##################################################
## V2
##################################################
'''
We use a standard deep RL setup for our agents. The agent’s policy and value functions are
parameterized by a convolutional neural network with 2 layers each of 32 output channels, followed
by two linear layers with 128 dimensions. Each of the layers are followed by ReLU activations. This
body then feeds two heads, a scalar value function and a softmax policy function over the five actions.
All of the CNN kernels are 3x3 with stride and padding of one.


We use a similar setup to that used in the Maze game. The architecture differences are that we have
an additional two convolutional layers at the beginning, use 256 output channels, and have output
dimensions of 1024 and 512, respectively, for the linear layers. This architecture was not tuned at all
during the course of our experiments. Further hyperparameter differences are that we used a learning
rate of 3 × 10−4
and a gamma of 1.0. These models trained for 72 hours, which is ∼50M frames.
'''
def BuildPPONetWorkProfileV2():
    network = list()
    
    # BUILD CNN
    network.append([
        dict(type='input', names='frame_feature'),
        dict(type='conv2d', size=256, window=3, stride=1, padding='SAME', activation='relu'),
        dict(type='pool2d', window=2, stride=2),
        dict(type='conv2d', size=256, window=3, stride=1, padding='SAME', activation='relu'),
        dict(type='pool2d', window=2, stride=2),
        dict(type='conv2d', size=32, window=3, stride=1, padding='SAME', activation='relu'),
        dict(type='pool2d', window=2, stride=2),
        dict(type='conv2d', size=32, window=3, stride=1, padding='SAME', activation='relu'),
        dict(type='pool2d', window=2, stride=2),
        dict(type='flatten'),
        dict(type='output', name='frame_feature-emb')
    ])
    
    # Final touch
    network.append([
        dict(type='input', names=['frame_feature-emb'],
             aggregation_type='concat'),
        dict(type='dense', size=1024, activation='relu'),
        dict(type='dense', size=512, activation='relu'),
    ])
    
    return network

def BuildBaseNetWorkProfileV2():
    network = list()
    
    # BUILD CNN
    network.append([
        dict(type='input', names='frame_feature'),
        dict(type='conv2d', size=256, window=3, stride=1, padding='SAME', activation='relu'),
        dict(type='pool2d', window=2, stride=2),
        #dict(type='conv2d', size=256, window=3, stride=1, padding='SAME', activation='relu'),
        #dict(type='pool2d', window=2, stride=2),
        #dict(type='conv2d', size=32, window=3, stride=1, padding='SAME', activation='relu'),
        #dict(type='pool2d', window=2, stride=2),
        dict(type='conv2d', size=32, window=3, stride=1, padding='SAME', activation='relu'),
        dict(type='pool2d', window=2, stride=2),
        dict(type='flatten'),
        dict(type='output', name='frame_feature-emb')
    ])
    
    # Final touch
    network.append([
        dict(type='input', names=['frame_feature-emb'],
             aggregation_type='concat'),
        dict(type='dense', size=512, activation='relu'),
        dict(type='dense', size=256, activation='relu'),
    ])
    return network
