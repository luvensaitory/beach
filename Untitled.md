

```python
import numpy as np
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense
import pandas as pd


df = pd.read_csv('input.csv',header=None)
#print (df)
X_train = df.as_matrix()
#print (a.shape)
df = pd.read_csv('output.csv',header=None)
y_train = df.as_matrix()
df = pd.read_csv('input_test.csv',header=None)
#print (df)
X_test = df.as_matrix()
#print (a.shape)
df = pd.read_csv('output_test.csv',header=None)
y_test = df.as_matrix()

X_train=X_train[:,np.newaxis,:]
X_test=X_test[:,np.newaxis,:]
```

    Using TensorFlow backend.
    


```python


#build model
model = Sequential()

model.add(SimpleRNN(batch_input_shape=(None, 1, 9), units = 50))
model.add(Dense(units=6, kernel_initializer='normal', activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#categorical_crossentropy
#優化器:sgd adagrad rmsprop adadelta adam adamax nadam tfoptimizer 
for step in range(1, 201):
    
    loss = model.train_on_batch(X_train, y_train)
    
    # 每 500 批，顯示測試的準確率
    # 模型評估
    loss, accuracy = model.evaluate(X_test, y_test, batch_size=y_test.shape[0], 
        verbose=False)
    print("test loss: {}  test accuracy: {}".format(loss,accuracy))

X = X_test
predictions = model.predict_classes(X)
# get prediction result

accuracy = 0

for i in range(10):
    for j in range(6):
        if y_test[i][j] == 1:
            ans = j
    if ans == predictions[i]:
        accuracy += 1

print(accuracy/28)
```

    test loss: 1.7753934860229492  test accuracy: 0.1785714328289032
    test loss: 1.755875587463379  test accuracy: 0.3214285671710968
    test loss: 1.7396243810653687  test accuracy: 0.3214285671710968
    test loss: 1.7252498865127563  test accuracy: 0.3214285671710968
    test loss: 1.7121154069900513  test accuracy: 0.3214285671710968
    test loss: 1.6998611688613892  test accuracy: 0.3571428656578064
    test loss: 1.6882588863372803  test accuracy: 0.3571428656578064
    test loss: 1.6771568059921265  test accuracy: 0.3571428656578064
    test loss: 1.6664505004882812  test accuracy: 0.3571428656578064
    test loss: 1.6560637950897217  test accuracy: 0.3571428656578064
    test loss: 1.6459404230117798  test accuracy: 0.3571428656578064
    test loss: 1.636036992073059  test accuracy: 0.3928571343421936
    test loss: 1.626320481300354  test accuracy: 0.3928571343421936
    test loss: 1.616763710975647  test accuracy: 0.3928571343421936
    test loss: 1.6073458194732666  test accuracy: 0.3928571343421936
    test loss: 1.5980497598648071  test accuracy: 0.3928571343421936
    test loss: 1.5888615846633911  test accuracy: 0.3928571343421936
    test loss: 1.5797702074050903  test accuracy: 0.4642857015132904
    test loss: 1.5707663297653198  test accuracy: 0.4642857015132904
    test loss: 1.561842679977417  test accuracy: 0.4642857015132904
    test loss: 1.5529930591583252  test accuracy: 0.4642857015132904
    test loss: 1.5442132949829102  test accuracy: 0.4642857015132904
    test loss: 1.5354993343353271  test accuracy: 0.4642857015132904
    test loss: 1.5268477201461792  test accuracy: 0.4642857015132904
    test loss: 1.5182565450668335  test accuracy: 0.4642857015132904
    test loss: 1.509724497795105  test accuracy: 0.4642857015132904
    test loss: 1.501250147819519  test accuracy: 0.4642857015132904
    test loss: 1.4928327798843384  test accuracy: 0.4642857015132904
    test loss: 1.4844721555709839  test accuracy: 0.4642857015132904
    test loss: 1.4761676788330078  test accuracy: 0.4642857015132904
    test loss: 1.4679194688796997  test accuracy: 0.4642857015132904
    test loss: 1.4597265720367432  test accuracy: 0.4642857015132904
    test loss: 1.45158851146698  test accuracy: 0.4642857015132904
    test loss: 1.4435045719146729  test accuracy: 0.4642857015132904
    test loss: 1.4354742765426636  test accuracy: 0.4642857015132904
    test loss: 1.4274966716766357  test accuracy: 0.4642857015132904
    test loss: 1.4195705652236938  test accuracy: 0.4642857015132904
    test loss: 1.4116954803466797  test accuracy: 0.4642857015132904
    test loss: 1.4038705825805664  test accuracy: 0.4642857015132904
    test loss: 1.3960953950881958  test accuracy: 0.4642857015132904
    test loss: 1.3883686065673828  test accuracy: 0.4642857015132904
    test loss: 1.380689263343811  test accuracy: 0.4642857015132904
    test loss: 1.3730555772781372  test accuracy: 0.4642857015132904
    test loss: 1.3654661178588867  test accuracy: 0.5
    test loss: 1.3579185009002686  test accuracy: 0.5
    test loss: 1.3504102230072021  test accuracy: 0.5357142686843872
    test loss: 1.3429381847381592  test accuracy: 0.5714285969734192
    test loss: 1.3354994058609009  test accuracy: 0.5714285969734192
    test loss: 1.3280903100967407  test accuracy: 0.5714285969734192
    test loss: 1.3207074403762817  test accuracy: 0.5714285969734192
    test loss: 1.3133474588394165  test accuracy: 0.5714285969734192
    test loss: 1.3060064315795898  test accuracy: 0.5714285969734192
    test loss: 1.2986814975738525  test accuracy: 0.5714285969734192
    test loss: 1.2913695573806763  test accuracy: 0.5714285969734192
    test loss: 1.2840681076049805  test accuracy: 0.5714285969734192
    test loss: 1.2767747640609741  test accuracy: 0.5714285969734192
    test loss: 1.269487977027893  test accuracy: 0.5714285969734192
    test loss: 1.2622060775756836  test accuracy: 0.5714285969734192
    test loss: 1.2549282312393188  test accuracy: 0.5714285969734192
    test loss: 1.247653841972351  test accuracy: 0.5714285969734192
    test loss: 1.240382194519043  test accuracy: 0.5714285969734192
    test loss: 1.2331135272979736  test accuracy: 0.5714285969734192
    test loss: 1.2258481979370117  test accuracy: 0.5714285969734192
    test loss: 1.2185863256454468  test accuracy: 0.5714285969734192
    test loss: 1.2113288640975952  test accuracy: 0.6071428656578064
    test loss: 1.2040765285491943  test accuracy: 0.6071428656578064
    test loss: 1.1968300342559814  test accuracy: 0.6071428656578064
    test loss: 1.1895904541015625  test accuracy: 0.6071428656578064
    test loss: 1.182358741760254  test accuracy: 0.6071428656578064
    test loss: 1.175135850906372  test accuracy: 0.6071428656578064
    test loss: 1.1679229736328125  test accuracy: 0.6071428656578064
    test loss: 1.1607210636138916  test accuracy: 0.6071428656578064
    test loss: 1.1535316705703735  test accuracy: 0.6071428656578064
    test loss: 1.1463555097579956  test accuracy: 0.6071428656578064
    test loss: 1.1391942501068115  test accuracy: 0.6071428656578064
    test loss: 1.132049322128296  test accuracy: 0.6071428656578064
    test loss: 1.1249219179153442  test accuracy: 0.6071428656578064
    test loss: 1.1178137063980103  test accuracy: 0.6428571343421936
    test loss: 1.110726237297058  test accuracy: 0.6428571343421936
    test loss: 1.103661060333252  test accuracy: 0.6428571343421936
    test loss: 1.0966198444366455  test accuracy: 0.6428571343421936
    test loss: 1.0896042585372925  test accuracy: 0.6785714030265808
    test loss: 1.082615852355957  test accuracy: 0.6785714030265808
    test loss: 1.0756564140319824  test accuracy: 0.6785714030265808
    test loss: 1.0687274932861328  test accuracy: 0.6785714030265808
    test loss: 1.0618304014205933  test accuracy: 0.6785714030265808
    test loss: 1.054966688156128  test accuracy: 0.6785714030265808
    test loss: 1.0481374263763428  test accuracy: 0.6785714030265808
    test loss: 1.0413439273834229  test accuracy: 0.6785714030265808
    test loss: 1.0345872640609741  test accuracy: 0.6785714030265808
    test loss: 1.0278680324554443  test accuracy: 0.6785714030265808
    test loss: 1.021187424659729  test accuracy: 0.6785714030265808
    test loss: 1.0145457983016968  test accuracy: 0.6785714030265808
    test loss: 1.0079439878463745  test accuracy: 0.6785714030265808
    test loss: 1.00138258934021  test accuracy: 0.6785714030265808
    test loss: 0.9948620200157166  test accuracy: 0.6785714030265808
    test loss: 0.988382875919342  test accuracy: 0.6785714030265808
    test loss: 0.9819455742835999  test accuracy: 0.6785714030265808
    test loss: 0.9755507111549377  test accuracy: 0.6785714030265808
    test loss: 0.9691984057426453  test accuracy: 0.6785714030265808
    test loss: 0.9628891348838806  test accuracy: 0.6785714030265808
    test loss: 0.9566229581832886  test accuracy: 0.7142857313156128
    test loss: 0.9504003524780273  test accuracy: 0.7142857313156128
    test loss: 0.9442213773727417  test accuracy: 0.7142857313156128
    test loss: 0.9380863904953003  test accuracy: 0.7857142686843872
    test loss: 0.9319957494735718  test accuracy: 0.7857142686843872
    test loss: 0.9259496927261353  test accuracy: 0.7857142686843872
    test loss: 0.9199486970901489  test accuracy: 0.7857142686843872
    test loss: 0.9139932990074158  test accuracy: 0.7857142686843872
    test loss: 0.9080837965011597  test accuracy: 0.7857142686843872
    test loss: 0.902220606803894  test accuracy: 0.7857142686843872
    test loss: 0.8964043855667114  test accuracy: 0.7857142686843872
    test loss: 0.8906356692314148  test accuracy: 0.7857142686843872
    test loss: 0.8849149942398071  test accuracy: 0.7857142686843872
    test loss: 0.8792431950569153  test accuracy: 0.7857142686843872
    test loss: 0.8736206889152527  test accuracy: 0.7857142686843872
    test loss: 0.8680484890937805  test accuracy: 0.7857142686843872
    test loss: 0.862527072429657  test accuracy: 0.7857142686843872
    test loss: 0.8570569157600403  test accuracy: 0.7857142686843872
    test loss: 0.8516384959220886  test accuracy: 0.7857142686843872
    test loss: 0.8462719917297363  test accuracy: 0.7857142686843872
    test loss: 0.8409573435783386  test accuracy: 0.7857142686843872
    test loss: 0.8356947302818298  test accuracy: 0.7857142686843872
    test loss: 0.8304837346076965  test accuracy: 0.7857142686843872
    test loss: 0.8253241181373596  test accuracy: 0.7857142686843872
    test loss: 0.8202158808708191  test accuracy: 0.7857142686843872
    test loss: 0.8151589035987854  test accuracy: 0.7857142686843872
    test loss: 0.8101528882980347  test accuracy: 0.7857142686843872
    test loss: 0.8051975965499878  test accuracy: 0.7857142686843872
    test loss: 0.8002930283546448  test accuracy: 0.7857142686843872
    test loss: 0.7954389452934265  test accuracy: 0.7857142686843872
    test loss: 0.7906351685523987  test accuracy: 0.7857142686843872
    test loss: 0.7858815789222717  test accuracy: 0.7857142686843872
    test loss: 0.7811776995658875  test accuracy: 0.7857142686843872
    test loss: 0.7765235900878906  test accuracy: 0.7857142686843872
    test loss: 0.771918773651123  test accuracy: 0.7857142686843872
    test loss: 0.7673630714416504  test accuracy: 0.8214285969734192
    test loss: 0.7628564834594727  test accuracy: 0.8214285969734192
    test loss: 0.7583985924720764  test accuracy: 0.8214285969734192
    test loss: 0.7539895176887512  test accuracy: 0.8214285969734192
    test loss: 0.7496289014816284  test accuracy: 0.8214285969734192
    test loss: 0.7453168034553528  test accuracy: 0.8214285969734192
    test loss: 0.74105304479599  test accuracy: 0.8214285969734192
    test loss: 0.7368375658988953  test accuracy: 0.8214285969734192
    test loss: 0.7326704859733582  test accuracy: 0.8214285969734192
    test loss: 0.7285513877868652  test accuracy: 0.8214285969734192
    test loss: 0.7244805097579956  test accuracy: 0.8214285969734192
    test loss: 0.7204574346542358  test accuracy: 0.8214285969734192
    test loss: 0.7164819836616516  test accuracy: 0.8214285969734192
    test loss: 0.7125459313392639  test accuracy: 0.8214285969734192
    test loss: 0.7086652517318726  test accuracy: 0.8214285969734192
    test loss: 0.7048316597938538  test accuracy: 0.8214285969734192
    test loss: 0.7010442018508911  test accuracy: 0.8214285969734192
    test loss: 0.6973026990890503  test accuracy: 0.8214285969734192
    test loss: 0.6936064958572388  test accuracy: 0.8214285969734192
    test loss: 0.6899552941322327  test accuracy: 0.8214285969734192
    test loss: 0.6863488554954529  test accuracy: 0.8214285969734192
    test loss: 0.6827866435050964  test accuracy: 0.8214285969734192
    test loss: 0.679268479347229  test accuracy: 0.8214285969734192
    test loss: 0.6757941842079163  test accuracy: 0.8214285969734192
    test loss: 0.6723631620407104  test accuracy: 0.8214285969734192
    test loss: 0.6689753532409668  test accuracy: 0.8214285969734192
    test loss: 0.665630042552948  test accuracy: 0.8214285969734192
    test loss: 0.6623215079307556  test accuracy: 0.8214285969734192
    test loss: 0.6590597033500671  test accuracy: 0.8214285969734192
    test loss: 0.6558400988578796  test accuracy: 0.8214285969734192
    test loss: 0.6526619791984558  test accuracy: 0.8214285969734192
    test loss: 0.6495245099067688  test accuracy: 0.8214285969734192
    test loss: 0.6464270949363708  test accuracy: 0.8214285969734192
    test loss: 0.6433693170547485  test accuracy: 0.8214285969734192
    test loss: 0.6403507590293884  test accuracy: 0.8214285969734192
    test loss: 0.6373706459999084  test accuracy: 0.8214285969734192
    test loss: 0.6344285607337952  test accuracy: 0.8214285969734192
    test loss: 0.6315239667892456  test accuracy: 0.8214285969734192
    test loss: 0.6286560297012329  test accuracy: 0.8214285969734192
    test loss: 0.6258243918418884  test accuracy: 0.8214285969734192
    test loss: 0.623028576374054  test accuracy: 0.8214285969734192
    test loss: 0.6202680468559265  test accuracy: 0.8214285969734192
    test loss: 0.6175424456596375  test accuracy: 0.8214285969734192
    test loss: 0.6148514747619629  test accuracy: 0.8214285969734192
    test loss: 0.6121947169303894  test accuracy: 0.8214285969734192
    test loss: 0.6095721125602722  test accuracy: 0.8214285969734192
    test loss: 0.6069831848144531  test accuracy: 0.8214285969734192
    test loss: 0.6044278740882874  test accuracy: 0.8214285969734192
    test loss: 0.6019057631492615  test accuracy: 0.8214285969734192
    test loss: 0.5994167327880859  test accuracy: 0.8214285969734192
    test loss: 0.5969604849815369  test accuracy: 0.8214285969734192
    test loss: 0.5945367813110352  test accuracy: 0.8214285969734192
    test loss: 0.5921453237533569  test accuracy: 0.8214285969734192
    test loss: 0.5897858738899231  test accuracy: 0.8214285969734192
    test loss: 0.5874578356742859  test accuracy: 0.8214285969734192
    test loss: 0.5851610898971558  test accuracy: 0.8214285969734192
    test loss: 0.5828949809074402  test accuracy: 0.8214285969734192
    test loss: 0.5806592702865601  test accuracy: 0.8214285969734192
    test loss: 0.5784533619880676  test accuracy: 0.8214285969734192
    test loss: 0.5762768983840942  test accuracy: 0.8214285969734192
    test loss: 0.5741294026374817  test accuracy: 0.8928571343421936
    test loss: 0.5720104575157166  test accuracy: 0.8928571343421936
    test loss: 0.569919228553772  test accuracy: 0.8928571343421936
    test loss: 0.5678561329841614  test accuracy: 0.8928571343421936
    0.2857142857142857
    
