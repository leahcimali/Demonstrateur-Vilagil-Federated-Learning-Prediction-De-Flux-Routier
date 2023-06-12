# Projet-Vilagil
## Application of Federated Learning to urban mobility data. 
### Associate to Toulouse city Vilagil Project.

Ongoing Project: Vilagil - Data and mobility.

Applications of Federated Learning to urban mobility data. 


https://www.banquedesterritoires.fr/sites/default/files/2020-11/Toulouse%2C%20Vilagil%20%28Occitanie%29.pdf

To install torch version, torchvision and torchaudio use this line :   
torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

### About Federated Learning

Federated learning can be used in different domains such as sentiment analysis, cell phone activity tracking and many other cases. Remote clients can be organizations or institutions such as hospitals, or individuals' mobile devices since they contain a multitude of data suitable for learning models such as images taken by the user, his location, notes, applications...

Before each training, only a fixed set of clients is allowed to participate for efficiency reasons, including those who are eligible based on certain heuristics. If we are testing on mobile devices for example, the eligible devices would be those: fully charged, with specific hardware configurations, connected to a reliable and free WiFi network and idle. This is in order to not interrupt the workout if the device is disconnected or runs out of battery, and to minimize the negative impact the local workout could have on the user's experience, e.g. workout time if the device is in use.

A model training is composed of several "rounds" (the number of times the initial model settings will be updated). For each round, only a fraction of the clients, whose number is fixed in advance, is used and may differ from round to round (e.g. 20% of the participating clients). 

At the beginning, the model is sent to the selected clients with the initial parameters (the weights of the neural network) which are set either by the server according to a strategy to be defined, or by a randomly chosen client.
Each client then performs a local calculation based on the global state of the shared model and its local dataset. The gradients are calculated and sent to the server in order to perform an aggregation and update it as the new global state and the process repeats for all rounds.


![](https://github.com/leahcimali/Federated-Traffic-Flow-Forecasting/blob/main/data/Fed1.png)

*FIG 1 : An illustration of Federated Learning Paradigm*

Communications between the server and the various clients must be secure [7], efficient, scalable and fault-tolerant, using the SSL (Secure Sockets Layer) protocol for encrypted exchanges between machines. Thus, the risks of eavesdropping or interception are reduced and the confidentiality and protection of user data is guaranteed.

![](https://github.com/leahcimali/Federated-Traffic-Flow-Forecasting/blob/main/data/Fed2.png)

*FIG 2 : A Federated Learning Architecture*

Unlike learning in traditional machine learning models, where train, test, and validation sets are obtained via explicit splits of the data, they are obtained in the case of federated learning by splitting the client devices into three distinct populations. The probability of reusing clients in the test, train or validation sets is very low in a sufficiently large client population.

Two approaches can be used for training a federated model  :

The same client participates in the train and the test, the local dataset will simply be split into two parts: some clients are used for the train, others for the test.

In the second approach, each client participating in the test, returns to the server a set of metrics according to its local dataset and the last version of the model received after the training.

Experiments results : https://drive.google.com/drive/folders/1SDk7oa4IaASBpCoj8BIQrStD3ixo-B2U?usp=drive_link
