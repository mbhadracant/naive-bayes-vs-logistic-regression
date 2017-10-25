import numpy as np

X = np.array([[6,180,12],[5.92,190,11],[5.58,170,12],[5.92,165,10],[5,100,6],[5.5,150,8],[5.42,130,7],[5.75,150,9]])
Y = np.array(['M','M','M','M','F','F','F','F'])

class NB:

    def train(self, data, labels):
        self.data = {}

        examples_size = len(data)
        if examples_size == 0:
            raise ValueError("the data provided is empty")


        features_size = len(data[0])
        self.data['examples_size'] = examples_size
        self.data['features_size'] = features_size

        distinct_labels = set(labels)
        self.data['calculations'] = {}

        for label in distinct_labels:
            self.data['calculations'][label] = {}
            self.data['calculations'][label]['probability'] = 0
            self.data['calculations'][label]['count'] = 0
            self.data['calculations'][label]['mean'] = [0] * features_size
            self.data['calculations'][label]['variance'] = [0] * features_size

        self.calculate_class_probabilities(labels)
        self.calculate_mean(data, labels)
        self.calculate_variance(data, labels)

    def calculate_class_probabilities(self, labels):

        for label in labels:
            self.data['calculations'][label]['count'] += 1

        for label in self.data['calculations']:
            self.data['calculations'][label]['probability'] = self.data['calculations'][label]['count'] / self.data['examples_size']



    def calculate_mean(self, data, labels):

        for i in range(0, self.data['features_size']):

            for j in range(0, self.data['examples_size']):
                self.data['calculations'][labels[j]]['mean'][i] += data[j][i]

            for label in self.data['calculations']:
                self.data['calculations'][label]['mean'][i] /= self.data['calculations'][label]['count']



    def calculate_variance(self, data, labels):

        for i in range(0, self.data['features_size']):

            for j in range(0, self.data['examples_size']):
                self.data['calculations'][labels[j]]['variance'][i] += np.power((data[j][i] - self.data['calculations'][labels[j]]['mean'][i]),2)

            for label in self.data['calculations']:
                self.data['calculations'][label]['variance'][i] /= self.data['calculations'][label]['count'] - 1


    def predict(self, data):

        if len(data) == 0:
            raise ValueError('Expecting 2D array')

        if len(data[0]) != self.data['features_size']:
            raise ValueError('Invalid feature size')

        predictions = []

        for example in data:
            posteriors = {}

            evidence = 0


            for label in self.data['calculations']:
                posteriors[label] = 1
                curr_evidence = self.data['calculations'][label]['probability']

                for i in range(0, self.data['features_size']):
                    variance = self.data['calculations'][label]['variance'][i]
                    mean = self.data['calculations'][label]['mean'][i]
                    numerator = (1/np.sqrt(2 * np.pi * variance)) * np.power(np.e, (-np.power((example[i] - mean),2)) / (2 * variance))
                    curr_evidence *= numerator
                    posteriors[label] *= numerator

                evidence += curr_evidence

            predicted_label = None
            highest_posterior = 0

            for label in posteriors:

                curr_posterior = posteriors[label] / evidence

                if curr_posterior > highest_posterior:
                    predicted_label = label
                    highest_posterior = curr_posterior

            predictions.append(predicted_label)

        return predictions

nb = NB()
nb.train(X,Y)

e = [[6,150,9]]

print(nb.predict(e))


