from sklearn.mixture import GaussianMixture
import numpy

# Test with multiple random seeds to find one that triggers the bug
for seed in range(100):
    numpy.random.seed(seed)
    X = numpy.random.randn(1000, 5)
    
    gm = GaussianMixture(n_components=5, n_init=5, random_state=seed)
    c1 = gm.fit_predict(X)
    c2 = gm.predict(X)
    
    if not numpy.array_equal(c1, c2):
        print(f'Bug found with seed={seed}!')
        print('Mismatch percentage:', numpy.sum(c1 != c2) / len(c1) * 100, '%')
        break
else:
    print('No bug found in 100 seeds')
