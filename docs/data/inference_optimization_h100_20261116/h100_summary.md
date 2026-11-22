
## graphene, e3nn eager: end_to_end median (ms)
| variant | baseline | host | tf32 | safe_scatter | compile_stages | compile | cudagraphs | compile_dynamic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deeph | 1045 | 95 | 94 | 94 | 98 | 97 | 102 | 107 |
| deeph_in_mandala | 1153 | 101 | 109 | 110 | 111 | 112 | 109 | 113 |
| deeph_e3 | 411 | 269 | 269 | 269 | 199 | 196 | 90 | 202 |
| deeph_e3_in_mandala | 547 | 279 | 278 | 279 | 213 | 214 | 95 | - |
| mace_h | 406 | 266 | 265 | 266 | 210 | 209 | 111 | 219 |
| mace_h_in_mandala | 549 | 282 | 281 | 282 | 228 | 229 | 120 | - |
| mandala | 549 | 527 | 518 | 528 | 450 | 450 | 317 | 446 |

## graphene, e3nn eager: end_to_end min (ms)
| variant | baseline | host | tf32 | safe_scatter | compile_stages | compile | cudagraphs | compile_dynamic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deeph | 1041 | 94 | 93 | 93 | 93 | 92 | 97 | 96 |
| deeph_in_mandala | 1151 | 101 | 101 | 101 | 107 | 107 | 105 | 104 |
| deeph_e3 | 410 | 268 | 268 | 267 | 199 | 195 | 89 | 201 |
| deeph_e3_in_mandala | 545 | 279 | 277 | 278 | 213 | 213 | 95 | - |
| mace_h | 405 | 265 | 264 | 265 | 209 | 208 | 111 | 218 |
| mace_h_in_mandala | 548 | 281 | 280 | 281 | 227 | 228 | 119 | - |
| mandala | 548 | 526 | 517 | 526 | 444 | 448 | 317 | 445 |

## graphene, e3nn eager: forward median (ms)
| variant | baseline | host | tf32 | safe_scatter | compile_stages | compile | cudagraphs | compile_dynamic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deeph | 6 | 6 | 5 | 6 | 4 | 4 | 3 | 5 |
| deeph_in_mandala | 284 | 21 | 21 | 21 | 20 | 20 | 19 | 22 |
| deeph_e3 | 235 | 235 | 235 | 236 | 165 | 163 | 55 | 168 |
| deeph_e3_in_mandala | 526 | 260 | 260 | 259 | 191 | 194 | 75 | - |
| mace_h | 233 | 233 | 232 | 233 | 177 | 176 | 78 | 185 |
| mace_h_in_mandala | 529 | 262 | 261 | 261 | 207 | 207 | 99 | - |
| mandala | 265 | 265 | 255 | 265 | 181 | 181 | 51 | 185 |

## graphene, e3nn eager: forward min (ms)
| variant | baseline | host | tf32 | safe_scatter | compile_stages | compile | cudagraphs | compile_dynamic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deeph | 6 | 6 | 5 | 6 | 4 | 4 | 3 | 5 |
| deeph_in_mandala | 283 | 21 | 20 | 21 | 20 | 20 | 19 | 21 |
| deeph_e3 | 234 | 234 | 234 | 235 | 164 | 162 | 55 | 168 |
| deeph_e3_in_mandala | 525 | 259 | 258 | 258 | 191 | 194 | 74 | - |
| mace_h | 232 | 232 | 232 | 233 | 176 | 176 | 78 | 184 |
| mace_h_in_mandala | 527 | 261 | 260 | 261 | 206 | 206 | 99 | - |
| mandala | 264 | 265 | 255 | 265 | 180 | 180 | 51 | 184 |

## graphene, e3nn eager: max |dH| / max |H| vs baseline (noise first)
| variant | noise | host | tf32 | safe_scatter | compile_stages | compile | cudagraphs | compile_dynamic | compile s (first pass, per stage) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| deeph | 7.0e-07 | 7.8e-07 | 4.4e-03 | 7.4e-07 | 9.0e-07 | 8.2e-07 | 9.0e-07 | 8.2e-07 | compile_stages=30, compile=2, cudagraphs=13, compile_dynamic=10 |
| deeph_in_mandala | 6.4e-07 | 7.1e-07 | 3.9e-03 | 6.8e-07 | 7.5e-07 | 7.5e-07 | 7.1e-07 | 7.1e-07 | compile_stages=7, compile=1, cudagraphs=9, compile_dynamic=5 |
| deeph_e3 | 5.6e-07 | 6.1e-07 | 1.9e-02 | 5.6e-07 | 6.5e-07 | 5.6e-07 | 6.1e-07 | 8.0e-07 | compile_stages=147, compile=19, cudagraphs=248, compile_dynamic=237 |
| deeph_e3_in_mandala | 5.0e-07 | 4.4e-07 | 1.4e-03 | 6.1e-07 | 5.0e-07 | 7.2e-07 | 6.1e-07 | - | compile_stages=160, compile=23, cudagraphs=252 |
| mace_h | 5.0e-07 | 6.4e-07 | 2.4e-02 | 6.0e-07 | 7.0e-07 | 7.4e-07 | 7.4e-07 | 8.3e-07 | compile_stages=189, compile=24, cudagraphs=342, compile_dynamic=180 |
| mace_h_in_mandala | 7.3e-07 | 6.3e-07 | 1.4e-03 | 6.8e-07 | 6.7e-07 | 6.7e-07 | 7.0e-07 | - | compile_stages=80, compile=28, cudagraphs=315 |
| mandala | 2.5e-07 | 1.6e-07 | 5.5e-04 | 1.6e-07 | 1.6e-07 | 1.6e-07 | 1.6e-07 | 1.6e-07 | compile_stages=318, compile=28, cudagraphs=960, compile_dynamic=328 |
- deeph: host patches identical {'from_packed': True, 'to_blocks': True, 'bond_frames@operator_features': True, 'bond_neighborhood_relations': True}
- deeph_in_mandala: host patches identical {'from_packed': True, 'trace_alignment': True, 'to_blocks': True, 'bond_frames@operator_features': True, 'bond_neighborhood_relations': True}
- deeph_e3: host patches identical {'from_packed': True, 'to_blocks': True}
- deeph_e3_in_mandala: host patches identical {'from_packed': True, 'trace_alignment': True, 'to_blocks': True}
- mace_h: host patches identical {'from_packed': True, 'to_blocks': True}
- mace_h_in_mandala: host patches identical {'from_packed': True, 'trace_alignment': True, 'to_blocks': True}
- mandala: host patches identical {'trace_alignment': True, 'to_blocks': True}

## mos2, e3nn eager: end_to_end median (ms)
| variant | baseline | host | tf32 | safe_scatter | compile_stages | compile | cudagraphs | compile_dynamic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deeph | 2011 | 307 | 314 | 313 | 314 | 316 | 319 | 323 |
| deeph_in_mandala | 2200 | 318 | 327 | 324 | 322 | 317 | 323 | 326 |
| deeph_e3 | 739 | 367 | 372 | 366 | 302 | 301 | 191 | 305 |
| deeph_e3_in_mandala | 913 | 385 | 390 | 384 | 320 | 317 | 197 | - |
| mace_h | 743 | 379 | 377 | 379 | 317 | 320 | 222 | 322 |
| mace_h_in_mandala | 931 | 399 | 398 | 399 | 337 | 336 | 232 | - |
| mandala | 701 | 673 | 663 | 674 | 592 | 595 | - | - |

## mos2, e3nn eager: end_to_end min (ms)
| variant | baseline | host | tf32 | safe_scatter | compile_stages | compile | cudagraphs | compile_dynamic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deeph | 1991 | 304 | 307 | 305 | 304 | 305 | 306 | 299 |
| deeph_in_mandala | 2155 | 314 | 318 | 317 | 309 | 313 | 314 | 320 |
| deeph_e3 | 735 | 366 | 372 | 366 | 300 | 298 | 189 | 303 |
| deeph_e3_in_mandala | 907 | 384 | 389 | 383 | 316 | 315 | 195 | - |
| mace_h | 737 | 378 | 376 | 378 | 316 | 317 | 221 | 321 |
| mace_h_in_mandala | 927 | 399 | 397 | 398 | 335 | 335 | 231 | - |
| mandala | 697 | 670 | 661 | 671 | 588 | 590 | - | - |

## mos2, e3nn eager: forward median (ms)
| variant | baseline | host | tf32 | safe_scatter | compile_stages | compile | cudagraphs | compile_dynamic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deeph | 7 | 7 | 6 | 7 | 5 | 5 | 4 | 6 |
| deeph_in_mandala | 593 | 66 | 67 | 66 | 66 | 65 | 65 | 67 |
| deeph_e3 | 239 | 239 | 241 | 239 | 172 | 172 | 60 | 173 |
| deeph_e3_in_mandala | 843 | 314 | 318 | 314 | 248 | 246 | 126 | - |
| mace_h | 252 | 253 | 247 | 252 | 190 | 191 | 96 | 195 |
| mace_h_in_mandala | 859 | 330 | 326 | 329 | 267 | 265 | 162 | - |
| mandala | 333 | 333 | 322 | 333 | 250 | 250 | - | - |

## mos2, e3nn eager: forward min (ms)
| variant | baseline | host | tf32 | safe_scatter | compile_stages | compile | cudagraphs | compile_dynamic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deeph | 6 | 6 | 6 | 7 | 5 | 5 | 3 | 6 |
| deeph_in_mandala | 589 | 66 | 67 | 66 | 65 | 65 | 64 | 66 |
| deeph_e3 | 239 | 239 | 240 | 239 | 170 | 169 | 59 | 173 |
| deeph_e3_in_mandala | 838 | 314 | 317 | 313 | 245 | 244 | 124 | - |
| mace_h | 252 | 252 | 247 | 252 | 189 | 190 | 95 | 194 |
| mace_h_in_mandala | 855 | 329 | 325 | 329 | 266 | 264 | 161 | - |
| mandala | 332 | 332 | 322 | 332 | 249 | 248 | - | - |

## mos2, e3nn eager: max |dH| / max |H| vs baseline (noise first)
| variant | noise | host | tf32 | safe_scatter | compile_stages | compile | cudagraphs | compile_dynamic | compile s (first pass, per stage) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| deeph | 6.4e-07 | 5.6e-07 | 4.7e-03 | 5.2e-07 | 7.0e-07 | 8.4e-07 | 8.6e-07 | 8.0e-07 | compile_stages=48, compile=4, cudagraphs=7, compile_dynamic=9 |
| deeph_in_mandala | 5.6e-07 | 5.6e-07 | 4.1e-03 | 6.0e-07 | 6.7e-07 | 6.7e-07 | 6.7e-07 | 6.3e-07 | compile_stages=13, compile=2, cudagraphs=4, compile_dynamic=6 |
| deeph_e3 | 4.4e-07 | 5.7e-07 | 2.6e-02 | 5.5e-07 | 5.1e-07 | 5.1e-07 | 6.4e-07 | 5.4e-07 | compile_stages=490, compile=56, cudagraphs=257, compile_dynamic=262 |
| deeph_e3_in_mandala | 7.7e-07 | 9.6e-07 | 1.3e-03 | 8.3e-07 | 7.8e-07 | 8.7e-07 | 9.3e-07 | - | compile_stages=446, compile=73, cudagraphs=266 |
| mace_h | 4.7e-07 | 4.9e-07 | 2.3e-02 | 4.3e-07 | 4.9e-07 | 4.5e-07 | 4.9e-07 | 5.1e-07 | compile_stages=274, compile=53, cudagraphs=141, compile_dynamic=229 |
| mace_h_in_mandala | 4.2e-07 | 5.0e-07 | 1.1e-03 | 5.3e-07 | 5.9e-07 | 6.3e-07 | 5.8e-07 | - | compile_stages=199, compile=65, cudagraphs=145 |
| mandala | 2.1e-07 | 2.1e-07 | 7.8e-04 | 2.1e-07 | 2.1e-07 | 2.1e-07 | - | - | compile_stages=756, compile=84 |
- deeph: host patches identical {'from_packed': True, 'to_blocks': True, 'bond_frames@operator_features': True, 'bond_neighborhood_relations': True}
- deeph_in_mandala: host patches identical {'from_packed': True, 'trace_alignment': True, 'to_blocks': True, 'bond_frames@operator_features': True, 'bond_neighborhood_relations': True}
- deeph_e3: host patches identical {'from_packed': True, 'to_blocks': True}
- deeph_e3_in_mandala: host patches identical {'from_packed': True, 'trace_alignment': True, 'to_blocks': True}
- mace_h: host patches identical {'from_packed': True, 'to_blocks': True}
- mace_h_in_mandala: host patches identical {'from_packed': True, 'trace_alignment': True, 'to_blocks': True}
- mandala: host patches identical {'trace_alignment': True, 'to_blocks': True}
