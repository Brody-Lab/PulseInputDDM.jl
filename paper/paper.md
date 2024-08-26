---
title: 'PulseInputDMM.jl: inference and learning for drift diffusions models fit with data from pulse-based evidence accumulation tasks'
tags:
  - Drift-diffusion models
  - Julia
authors:
  - name: Brian D. DePasquale
    orcid: 0000-0003-4508-0537
    affiliation: "1,2"
  - name: Diksha Gupta
    affiliation: "1"
  - name: Alex Piet
    affiliation: "1"
  - name: Thomas Luo
    affiliation: "1"
  - name: Tim Kim
    affiliation: "1"
  - name: Jorge Yanar
    affiliation: "1"
  - name: Emily Dennis
    affiliation: "1"
  - name: Marino Pagan
    affiliation: "1"
  - name: Chuck Kopec
    affiliation: "1"
  - name: Tyler Boyd-Meredith
    affiliation: "1"
  - name: Bing Brunton
    affiliation: "1"
  - name: Jonathan Pillow
    affiliation: "1"
  - name: Carlos D. Brody
    affiliation: "1"

affiliations:
 - name: Princeton Neuroscience Institute, Princeton University
   index: 1
 - name: Department of Biomedical Engineering, Boston University
   index: 2
date: 5 August 2024

bibliography: paper.bib
---

# Summary

Drift diffusion models (DDMs) are a popular model class for modeling a unobserved process that determines an subject's choice during a decision-making task [@Bogacz2006]. 

```math
 dz = \lambda zdt + u(t)dt + \sigma dW
```

[@Brunton2013]. 

# Statement of need

The initial motivation for writing ``PulseInputDDM.jl`` was to analyze experimental data collected from rats performing pulse-based evidence accumulation tasks. These findings were published in [@DePasquale2024]. 

![Model](fig1.png)

`PyDDM` [@PyDDM2020]

# Package design

# Example

# Availability

# Conclusion

``PulseInputDDM.jl`` is publicly available under the [MIT license](https://github.com/Brody-Lab/PulseInputDDM.jl/blob/master/LICENSE) at <https://github.com/Brody-Lab/PulseInputDDM.jl>.

# Author contributions

BD did XXX. BB did XXX.

# Acknowledgements

This work was supported by the Princeton Neuroscience Institute and the Simons Foundation.

# References
