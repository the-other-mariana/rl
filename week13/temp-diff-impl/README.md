# Bellman's Optimality Equations: Temporary Difference

- v(s) version [here](./v-temp-diff.py)

    - Deterministic 

    ```
    Optimal politic so far:
    sf1 = <-,       s1 = ->,        s2 = ->,        s3 = ->,        sf2 = ->,
    ==========================
    Iteration 6, cycle 0:
    fr: [-10, 0, -0.4, -0.4, 10]
    sf1     s1      s2      s3      sf2
    V(s) final
    0.79    -0.20   0.08    8.40    8.38
    Optimal politic so far:
    sf1 = <-,       s1 = ->,        s2 = ->,        s3 = ->,        sf2 = ->,
    ```

    - Non-deterministic

    ```
    Optimal politic so far:
    sf1 = ->,       s1 = ->,        s2 = <-,        s3 = <-,        sf2 = <-,
    Optimal politic so far:
    sf1 = ->,       s1 = ->,        s2 = ->,        s3 = <-,        sf2 = <-,
    Optimal politic so far:
    sf1 = ->,       s1 = ->,        s2 = ->,        s3 = ->,        sf2 = ->,
    ==========================
    Iteration 9, cycle 2:
    fr: [-10, 0, -0.4, -0.4, 10]
    sf1     s1      s2      s3      sf2
    V(s) final
    -3.91   -3.08   -0.45   6.65    7.65
    Optimal politic so far:
    sf1 = ->,       s1 = ->,        s2 = ->,        s3 = ->,        sf2 = ->,
    ```

- q(s, a) version [here](./q-temp-diff.py)

    - Deterministic

    ```
    Optimal politic so far:
    sf1 = ->,       s1 = <-,        s2 = <-,        s3 = <-,        sf2 = <-,
    Optimal politic so far:
    sf1 = ->,       s1 = ->,        s2 = ->,        s3 = ->,        sf2 = ->,
    ==========================
    Iteration 12, cycle 1:
    fr: [-10, 0, -0.4, -0.4, 10]
    sf1     s1      s2      s3      sf2
    Q(s,a) final
    0.20    -0.32   -0.09   8.46    8.51
    0.26    -7.42   0.03    -0.13   0.28
    Optimal politic so far:
    sf1 = ->,       s1 = ->,        s2 = ->,        s3 = ->,        sf2 = ->,
    ```

    - Non-deterministic

    ```
    Optimal politic so far:
    sf1 = ->,       s1 = ->,        s2 = <-,        s3 = <-,        sf2 = <-,
    Optimal politic so far:
    sf1 = <-,       s1 = <-,        s2 = <-,        s3 = <-,        sf2 = <-,
    Optimal politic so far:
    sf1 = <-,       s1 = ->,        s2 = ->,        s3 = ->,        sf2 = ->,
    Optimal politic so far:
    sf1 = ->,       s1 = ->,        s2 = ->,        s3 = ->,        sf2 = ->,
    ==========================
    Iteration 12, cycle 3:
    fr: [-10, 0, -0.4, -0.4, 10]
    sf1     s1      s2      s3      sf2
    Q(s,a) final
    -1.85   -1.78   4.80    7.71    6.82
    -5.88   -6.00   0.10    0.46    0.39
    Optimal politic so far:
    sf1 = ->,       s1 = ->,        s2 = ->,        s3 = ->,        sf2 = ->,
    ```