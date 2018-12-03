from tensorforce.core.networks import Network
import tensorflow as tf


class PommerNet(Network):
    def __init__(self, scope='network', summary_labels=None):
        super().__init__(scope, summary_labels)
        
        self.map_size=[11, 11, 1]
    
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
        board=tf.reshape(ob[:121], self.map_size)
        bomb_blast_strength=tf.reshape(ob[121:242], self.map_size)
        bomb_life=tf.reshape(ob[242:363], self.map_size)
        position=ob[363:365]
        ammo=ob[365]
        blast_strength=ob[366]
        can_kick=ob[367]
        teammate=ob[368]
        enemies=ob[369:]
        
        assert tf.shape(ob)[-1]==372
        return {
            'board': board,
            'bomb_blast_strength': bomb_blast_strength,
            'bomb_life': bomb_life,
            'position': position,
            'ammo': ammo,
            'blast_strength': blast_strength,
            'can_kick': can_kick,
            'teammate': teammate,
            'enemies': enemies,
        }
    
    def build_cnn(self, image, name, initializer):
        # CNN, learnt from DQN from Atari
        weights = tf.get_variable(name=name+'_W1', shape=(3, 3, 3, 1), initializer=initializer)
        image = tf.nn.conv2d(image, filter=weights, strides=(1, 1, 1, 1), padding='SAME')
        image = tf.nn.relu(image)
        image = tf.nn.max_pool(image, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

        weights = tf.get_variable(name=name+'_W2', shape=(3, 3, 3, 1), initializer=initializer)
        image = tf.nn.conv2d(image, filter=weights, strides=(1, 1, 1, 1), padding='SAME')
        image = tf.nn.relu(image)
        image = tf.nn.max_pool(image, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

        image = tf.reshape(image, shape=(-1))
        image = tf.reduce_mean(image, axis=1)
        return image
        
    def tf_apply(self, x, internals, update, return_internals=False):
        observation = x['state']
        features = self.parse_observation(observation)
        
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32)

        board = self.build_cnn(features['board'], board, initializer)
        bbs = self.build_cnn(features['board'], board, initializer)
        blife = self.build_cnn(features['board'], board, initializer)
        
        # Combination        
        x = tf.concate(1, [board, bbs, blife, 
                               features['position'], 
                               features['ammo'], 
                               features['blast_strength'], 
                               features['can_kick'], 
                               features['teammate'], # Change to a better way..
                               features['enemies']])
        
        x = tf.layers.dense(x, 64, tf.nn.relu6)
        x = tf.layers.dense(x, 64, tf.nn.relu6)
        
        if return_internals:
            return x, list()
        else:
            return x

PommerNetBase=PommerNet