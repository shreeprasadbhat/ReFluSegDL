class DiceScore(tf.keras.metrics.Metric):
    
    def __init__(self, idx, name="dice_score", **kwargs):
        super(DiceScore, self).__init__(name=name, **kwargs)

        # accumulate numerator and denominator over batches of volume(NUM_BSCANS_PER_VOLUME)
        self.nom =  self.add_weight(shape=(OUTPUT_CHANNELS,), name='nom', initializer='zeros') 
        self.denom = self.add_weight(shape=(OUTPUT_CHANNELS,), name='denom', initializer='zeros')

        self.dice_score = self.add_weight(shape=(OUTPUT_CHANNELS,), name='dice_score', initializer='zeros')
        self.idx = idx
        

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_pred = tf.one_hot(tf.argmax(y_pred, axis=-1), OUTPUT_CHANNELS)
        y_true = tf.one_hot(tf.cast(tf.squeeze(y_true, axis=-1), tf.int64), OUTPUT_CHANNELS, axis=-1)

        self.nom.assign_add(tf.reduce_sum(y_true*y_pred, axis=(0,1,2)))
        self.denom.assign_add(tf.reduce_sum(y_true, axis=(0,1,2)) + tf.reduce_sum(y_pred, axis=(0,1,2)))


    def result(self):
        dice_score = (2. * self.nom[..., self.idx]) / (self.denom[..., self.idx])
        return 1. if tf.math.is_nan(dice_score) else dice_score

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.nom.assign_sub(self.nom)
        self.denom.assign_sub(self.denom)
        self.dice_score.assign_sub(self.dice_score)
