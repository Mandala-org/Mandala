Radial basis: ``bessel_basis``
==============================

Role
----

A radial basis turns an interatomic distance into a vector of smooth
invariant features :math:`\bm\varphi(r)`. Every distance dependence in the
network — radial filters of convolutions, attention scores, encoder
amplitudes — is a learned function of these features, so the basis sets the
radial resolution of the model.

Built-in members of this kind: ``soft_one_hot_basis`` (the e3nn bases; the
Gaussian one is MANDALA's default), ``gaussian_basis`` (fixed or trainable
centres) and the sine basis with polynomial cutoff ``cutoff_radial_expansion``.

The new layer
-------------

Spherical Bessel functions of order zero are the eigenfunctions of the
Laplacian in a ball with Dirichlet boundary at :math:`r_c`:

.. math::

   \varphi_n(r)=\sqrt{\frac{2}{r_c}}\;\frac{\sin(n\pi r/r_c)}{r}\;u\!\left(\frac{r}{r_c}\right),
   \qquad n=1,\dots,N .

They are orthogonal on :math:`[0,r_c]` with weight :math:`r^2`, so different
basis functions carry non-redundant radial information. The polynomial
envelope

.. math::

   u(x)=1-\frac{(p+1)(p+2)}{2}x^{p}+p(p+2)x^{p+1}-\frac{p(p+1)}{2}x^{p+2},
   \qquad u(1)=u'(1)=u''(1)=0,

brings every feature smoothly to zero at the cutoff, so the predicted
operator varies smoothly when an atom enters or leaves the neighbourhood —
important for molecular dynamics and for differentiable geometry. Optional
trainable frequencies let the model adapt the radial resolution.

**Onsite edges.** At :math:`r=0`, :math:`\sin(n\pi r/r_c)/r` is finite
(:math:`n\pi/r_c`) but its direct evaluation divides by zero. Writing it as

.. math::

   \frac{\sin(n\pi r/r_c)}{r}=\frac{n\pi}{r_c}\,\operatorname{sinc}\!\left(\frac{nr}{r_c}\right),
   \qquad \operatorname{sinc}(x)=\frac{\sin\pi x}{\pi x},

gives exact values and gradients everywhere (checklist item 7).

**Symmetry.** Lengths are invariant, so any function of them is invariant.

Implementation
--------------

.. literalinclude:: ../../examples/layer_cookbook/layers.py
   :language: python
   :pyobject: BesselBasis

The factory requires one length per edge and declares ``N`` invariant
scalars on the same support:

.. literalinclude:: ../../examples/layer_cookbook/layers.py
   :language: python
   :pyobject: _bessel_basis

Usage
-----

.. code-block:: yaml

   - id: radial
     transform: bessel_basis
     inputs: [edge_lengths]
     outputs: [radial]
     options: {cutoff: 6.0, n_basis: 8, envelope_exponent: 6, trainable: false}

``radial`` is a ``8x0e`` edge feature and can be consumed by any layer that
expects invariant radial features, including the built-in
``radial_channel_scale`` and ``projected_attention``.

Variations
----------

* Replace the envelope by a learned monotone function to let the network
  choose its effective range.
* Scale the argument per species pair by a characteristic bond length
  :math:`r_0^{(ab)}` (as ``pair_distance_normalization: pair_r0`` does for the
  Gaussian basis) to give every bond type the same resolution near its
  equilibrium distance.
