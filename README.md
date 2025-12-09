# Memory-augmented Variational Recurrent Networks (mVRNN) 

### Requirements

- python 3
- pytorch >= 1.0.0
- Gensim
- sklearn
- numpy
- json
- pandas
- math

### Usage

#### Time series prediction

  For example, you can run the following code to train an `mVRNN` model on the `DJI` dataset.

```
  python train.py --dataset 'DJI' --algorithm 'mVRNN'
```

**Available stock volatility datasets**

- Dow Jones Industrial Average: `DJI`
- Nasdaq Composite: `NSDQ`
- Apple Inc.: `AAPL`
- Google: `GOOGL`

**Available algorithms**

- vanilla RNN: `RNN`
- vanilla LSTM: `LSTM`
- Memory-augmented RNN with homogeneous memory parameter d: `mRNN_fixD`
- Memory-augmented LSTM with homogeneous d: `mLSTM_fixD`
- Memory-augmented Variational RNN with homogeneous memory parameter d: `mVRNN_fixD`
- Memory-augmented Variational RNN with dynamic memory parameter d: `mVRNN`
- Memory-augmented Variational RNN with Wasserstein autoencoder and homogenous memory parameter d: `mVRNN_fixD_WAE`
- Memory-augmented Variational RNN with Wasserstein autoencoder and dynamic memory parameter d: `mVRNN_WAE`