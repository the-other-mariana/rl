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