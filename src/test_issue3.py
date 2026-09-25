from sklearn.mixture import GaussianMixture
import numpy

# Create data with clear cluster structure
numpy.random.seed(42)
n_samples = 500
X1 = numpy.random.randn(n_samples, 2) + [5, 5]
X2 = numpy.random.randn(n_samples, 2) + [-5, -5]
X3 = numpy.random.randn(n_samples, 2) + [5, -5]
X4 = numpy.random.randn(n_samples, 2) + [-5, 5]
X5 = numpy.random.randn(n_samples, 2)
X = numpy.vstack([X1, X2, X3, X4, X5])

print('Testing with n_init=10')
for seed in range(20):
    gm = GaussianMixture(n_components=5, n_init=10, random_state=seed)
    c1 = gm.fit_predict(X)
    c2 = gm.predict(X)
    
    if not numpy.array_equal(c1, c2):
        print(f'Bug found with seed={seed}!')
        print('Mismatch percentage:', numpy.sum(c1 != c2) / len(c1) * 100, '%')
        break
else:
    print('No bug found in 20 seeds')
