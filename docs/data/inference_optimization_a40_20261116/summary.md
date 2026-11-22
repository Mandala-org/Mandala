
## graphene, e3nn eager: end_to_end median (ms)
| variant | baseline | host | tf32 | safe_scatter | compile_stages | compile | cudagraphs | compile_dynamic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deeph | 1662 | 185 | 187 | 187 | 200 | 180 | 180 | 206 |
| deeph_in_mandala | 1970 | 202 | 195 | 187 | 200 | 214 | 206 | 219 |
| deeph_e3 | 603 | 407 | 405 | 373 | 294 | 287 | 144 | 294 |
| deeph_e3_in_mandala | 825 | 410 | 422 | 418 | 272 | 307 | 159 | - |
| mace_h | 683 | 495 | 487 | 491 | 353 | 347 | 252 | 358 |
| mace_h_in_mandala | 874 | 519 | 527 | 504 | 351 | 357 | 255 | - |
| mandala | 743 | 721 | 727 | 692 | 573 | 555 | 419 | 564 |

## graphene, e3nn eager: end_to_end min (ms)
| variant | baseline | host | tf32 | safe_scatter | compile_stages | compile | cudagraphs | compile_dynamic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deeph | 1614 | 175 | 175 | 161 | 179 | 168 | 169 | 187 |
| deeph_in_mandala | 1836 | 176 | 175 | 178 | 189 | 199 | 194 | 208 |
| deeph_e3 | 583 | 391 | 373 | 361 | 287 | 279 | 141 | 286 |
| deeph_e3_in_mandala | 763 | 395 | 411 | 404 | 267 | 300 | 156 | - |
| mace_h | 678 | 490 | 482 | 487 | 343 | 333 | 247 | 343 |
| mace_h_in_mandala | 845 | 504 | 518 | 497 | 340 | 347 | 248 | - |
| mandala | 715 | 686 | 675 | 669 | 542 | 538 | 410 | 539 |

## graphene, e3nn eager: forward median (ms)
| variant | baseline | host | tf32 | safe_scatter | compile_stages | compile | cudagraphs | compile_dynamic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deeph | 14 | 14 | 12 | 15 | 9 | 9 | 8 | 11 |
| deeph_in_mandala | 422 | 34 | 31 | 35 | 31 | 31 | 30 | 33 |
| deeph_e3 | 342 | 358 | 354 | 327 | 236 | 233 | 99 | 236 |
| deeph_e3_in_mandala | 761 | 376 | 394 | 386 | 241 | 280 | 128 | - |
| mace_h | 447 | 451 | 445 | 447 | 295 | 300 | 199 | 300 |
| mace_h_in_mandala | 840 | 479 | 497 | 492 | 328 | 324 | 221 | - |
| mandala | 414 | 389 | 409 | 387 | 247 | 248 | 112 | 256 |

## graphene, e3nn eager: forward min (ms)
| variant | baseline | host | tf32 | safe_scatter | compile_stages | compile | cudagraphs | compile_dynamic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deeph | 14 | 14 | 12 | 14 | 9 | 8 | 8 | 10 |
| deeph_in_mandala | 407 | 32 | 30 | 33 | 29 | 29 | 28 | 30 |
| deeph_e3 | 332 | 345 | 345 | 321 | 231 | 225 | 97 | 227 |
| deeph_e3_in_mandala | 735 | 364 | 380 | 378 | 237 | 270 | 125 | - |
| mace_h | 444 | 447 | 440 | 442 | 282 | 292 | 197 | 291 |
| mace_h_in_mandala | 822 | 470 | 484 | 469 | 318 | 310 | 217 | - |
| mandala | 389 | 377 | 389 | 376 | 237 | 238 | 110 | 250 |

## graphene, e3nn eager: max |dH| / max |H| vs baseline (noise first)
| variant | noise | host | tf32 | safe_scatter | compile_stages | compile | cudagraphs | compile_dynamic | compile s (first pass, per stage) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| deeph | 6.6e-07 | 7.4e-07 | 4.5e-03 | 7.0e-07 | 9.0e-07 | 9.8e-07 | 9.4e-07 | 8.2e-07 | compile_stages=9, compile=2, cudagraphs=15, compile_dynamic=12 |
| deeph_in_mandala | 5.7e-07 | 6.4e-07 | 3.7e-03 | 6.8e-07 | 7.7e-07 | 8.2e-07 | 8.6e-07 | 8.2e-07 | compile_stages=4, compile=1, cudagraphs=17, compile_dynamic=6 |
| deeph_e3 | 4.7e-07 | 5.6e-07 | 6.2e-04 | 6.5e-07 | 6.8e-07 | 5.8e-07 | 6.6e-07 | 6.1e-07 | compile_stages=170, compile=30, cudagraphs=431, compile_dynamic=306 |
| deeph_e3_in_mandala | 5.0e-07 | 5.5e-07 | 8.8e-04 | 5.3e-07 | 6.1e-07 | 6.1e-07 | 5.3e-07 | - | compile_stages=165, compile=35, cudagraphs=439 |
| mace_h | 5.8e-07 | 6.0e-07 | 6.6e-04 | 7.4e-07 | 7.0e-07 | 7.2e-07 | 6.8e-07 | 7.9e-07 | compile_stages=104, compile=36, cudagraphs=590, compile_dynamic=208 |
| mace_h_in_mandala | 5.3e-07 | 6.3e-07 | 8.9e-04 | 7.3e-07 | 7.3e-07 | 8.0e-07 | 8.0e-07 | - | compile_stages=82, compile=42, cudagraphs=572 |
| mandala | 2.5e-07 | 2.5e-07 | 3.4e-04 | 2.5e-07 | 2.5e-07 | 2.5e-07 | 2.5e-07 | 2.5e-07 | compile_stages=232, compile=40, cudagraphs=1649, compile_dynamic=358 |
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
| deeph | 2845 | 466 | 461 | 469 | 463 | 450 | 448 | 486 |
| deeph_in_mandala | 3255 | 464 | 472 | 478 | 481 | 479 | 475 | 485 |
| deeph_e3 | 948 | 512 | 526 | 519 | 404 | 423 | 299 | 413 |
| deeph_e3_in_mandala | 1188 | 530 | 533 | 533 | 449 | 445 | - | - |
| mace_h | 1102 | 658 | 660 | 665 | 518 | 523 | - | - |
| mace_h_in_mandala | 1329 | 689 | 689 | 691 | 549 | 559 | - | - |
| mandala | 887 | 853 | 867 | 862 | 753 | 760 | - | - |

## mos2, e3nn eager: end_to_end min (ms)
| variant | baseline | host | tf32 | safe_scatter | compile_stages | compile | cudagraphs | compile_dynamic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deeph | 2778 | 451 | 445 | 440 | 437 | 424 | 435 | 452 |
| deeph_in_mandala | 3040 | 452 | 432 | 466 | 447 | 465 | 460 | 471 |
| deeph_e3 | 937 | 499 | 514 | 510 | 396 | 409 | 291 | 405 |
| deeph_e3_in_mandala | 1161 | 515 | 526 | 519 | 422 | 415 | - | - |
| mace_h | 1092 | 650 | 652 | 658 | 513 | 511 | - | - |
| mace_h_in_mandala | 1307 | 682 | 681 | 680 | 532 | 531 | - | - |
| mandala | 866 | 842 | 844 | 843 | 739 | 742 | - | - |

## mos2, e3nn eager: forward median (ms)
| variant | baseline | host | tf32 | safe_scatter | compile_stages | compile | cudagraphs | compile_dynamic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deeph | 18 | 19 | 15 | 19 | 11 | 11 | 10 | 13 |
| deeph_in_mandala | 774 | 96 | 94 | 97 | 91 | 92 | 91 | 92 |
| deeph_e3 | 339 | 331 | 336 | 334 | 223 | 237 | 116 | 230 |
| deeph_e3_in_mandala | 1095 | 431 | 436 | 432 | 341 | 343 | - | - |
| mace_h | 485 | 481 | 481 | 483 | 338 | 341 | - | - |
| mace_h_in_mandala | 1225 | 590 | 587 | 586 | 443 | 423 | - | - |
| mandala | 487 | 490 | 494 | 493 | 365 | 362 | - | - |

## mos2, e3nn eager: forward min (ms)
| variant | baseline | host | tf32 | safe_scatter | compile_stages | compile | cudagraphs | compile_dynamic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deeph | 18 | 18 | 15 | 18 | 11 | 11 | 10 | 12 |
| deeph_in_mandala | 746 | 95 | 92 | 95 | 90 | 90 | 89 | 90 |
| deeph_e3 | 331 | 328 | 331 | 328 | 217 | 231 | 114 | 225 |
| deeph_e3_in_mandala | 1077 | 424 | 428 | 426 | 329 | 333 | - | - |
| mace_h | 479 | 478 | 478 | 478 | 328 | 334 | - | - |
| mace_h_in_mandala | 1206 | 583 | 582 | 577 | 435 | 419 | - | - |
| mandala | 472 | 481 | 480 | 481 | 358 | 356 | - | - |

## mos2, e3nn eager: max |dH| / max |H| vs baseline (noise first)
| variant | noise | host | tf32 | safe_scatter | compile_stages | compile | cudagraphs | compile_dynamic | compile s (first pass, per stage) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| deeph | 5.6e-07 | 5.6e-07 | 4.6e-03 | 5.6e-07 | 7.6e-07 | 7.6e-07 | 8.6e-07 | 9.4e-07 | compile_stages=19, compile=5, cudagraphs=9, compile_dynamic=4 |
| deeph_in_mandala | 4.8e-07 | 5.2e-07 | 4.4e-03 | 6.3e-07 | 6.7e-07 | 6.3e-07 | 6.3e-07 | 6.3e-07 | compile_stages=10, compile=2, cudagraphs=5, compile_dynamic=4 |
| deeph_e3 | 4.0e-07 | 4.4e-07 | 6.7e-04 | 5.1e-07 | 6.2e-07 | 6.4e-07 | 5.2e-07 | 5.6e-07 | compile_stages=477, compile=81, cudagraphs=356, compile_dynamic=175 |
| deeph_e3_in_mandala | 7.3e-07 | 7.1e-07 | 1.0e-03 | 8.7e-07 | 8.3e-07 | 8.0e-07 | - | - | compile_stages=439, compile=112 |
| mace_h | 3.8e-07 | 3.8e-07 | 5.8e-04 | 5.1e-07 | 4.7e-07 | 5.0e-07 | - | - | compile_stages=209, compile=80 |
| mace_h_in_mandala | 4.0e-07 | 5.3e-07 | 6.7e-04 | 6.3e-07 | 5.5e-07 | 5.8e-07 | - | - | compile_stages=172, compile=96 |
| mandala | 2.1e-07 | 2.1e-07 | 4.8e-04 | 2.1e-07 | 2.1e-07 | 2.1e-07 | - | - | compile_stages=644, compile=119 |
- deeph: host patches identical {'from_packed': True, 'to_blocks': True, 'bond_frames@operator_features': True, 'bond_neighborhood_relations': True}
- deeph_in_mandala: host patches identical {'from_packed': True, 'trace_alignment': True, 'to_blocks': True, 'bond_frames@operator_features': True, 'bond_neighborhood_relations': True}
- deeph_e3: host patches identical {'from_packed': True, 'to_blocks': True}
- deeph_e3_in_mandala: host patches identical {'from_packed': True, 'trace_alignment': True, 'to_blocks': True}
- mace_h: host patches identical {'from_packed': True, 'to_blocks': True}
- mace_h_in_mandala: host patches identical {'from_packed': True, 'trace_alignment': True, 'to_blocks': True}
- mandala: host patches identical {'trace_alignment': True, 'to_blocks': True}
