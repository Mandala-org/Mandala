
## graphene, e3nn eager: end_to_end median (ms)
| variant | baseline | host | tf32 | safe_scatter | compile_stages | compile | cudagraphs |
|---|---:|---:|---:|---:|---:|---:|---:|
| deeph | 1661 | 165 | 162 | 167 | 172 | 174 | 166 |
| deeph_in_mandala | 1855 | 172 | 176 | 180 | 186 | 184 | 181 |
| deeph_e3 | 566 | 374 | 376 | 376 | 271 | 266 | 141 |
| deeph_e3_in_mandala | 720 | 356 | 355 | 386 | 283 | 290 | 159 |
| mace_h | 681 | 514 | 496 | 490 | 336 | 334 | 231 |
| mace_h_in_mandala | 998 | 512 | 507 | 545 | 382 | 373 | 243 |
| mandala | 665 | err | 644 | 638 | 581 | 559 | 455 |

## graphene, e3nn eager: end_to_end min (ms)
| variant | baseline | host | tf32 | safe_scatter | compile_stages | compile | cudagraphs |
|---|---:|---:|---:|---:|---:|---:|---:|
| deeph | 1629 | 156 | 158 | 160 | 165 | 162 | 159 |
| deeph_in_mandala | 1793 | 162 | 166 | 168 | 175 | 174 | 167 |
| deeph_e3 | 554 | 363 | 366 | 358 | 262 | 254 | 139 |
| deeph_e3_in_mandala | 694 | 343 | 340 | 364 | 264 | 280 | 156 |
| mace_h | 673 | 503 | 482 | 485 | 325 | 325 | 230 |
| mace_h_in_mandala | 860 | 501 | 497 | 513 | 364 | 351 | 240 |
| mandala | 646 | err | 636 | 627 | 552 | 541 | 406 |

## graphene, e3nn eager: forward median (ms)
| variant | baseline | host | tf32 | safe_scatter | compile_stages | compile | cudagraphs |
|---|---:|---:|---:|---:|---:|---:|---:|
| deeph | 14 | 14 | 12 | 14 | 9 | 9 | 8 |
| deeph_in_mandala | 413 | 32 | 31 | 32 | 29 | 29 | 28 |
| deeph_e3 | 331 | 334 | 329 | 337 | 231 | 231 | 96 |
| deeph_e3_in_mandala | 699 | 342 | 326 | 329 | 260 | 260 | 129 |
| mace_h | 450 | 483 | 447 | 446 | 291 | 288 | 190 |
| mace_h_in_mandala | 853 | 492 | 497 | 512 | 351 | 335 | 220 |
| mandala | 345 | err | 340 | 333 | 232 | 239 | 107 |

## graphene, e3nn eager: forward min (ms)
| variant | baseline | host | tf32 | safe_scatter | compile_stages | compile | cudagraphs |
|---|---:|---:|---:|---:|---:|---:|---:|
| deeph | 14 | 14 | 12 | 14 | 9 | 9 | 8 |
| deeph_in_mandala | 395 | 32 | 30 | 32 | 29 | 28 | 27 |
| deeph_e3 | 315 | 324 | 324 | 322 | 216 | 213 | 91 |
| deeph_e3_in_mandala | 662 | 331 | 317 | 320 | 222 | 244 | 127 |
| mace_h | 443 | 464 | 440 | 441 | 285 | 281 | 186 |
| mace_h_in_mandala | 831 | 474 | 468 | 478 | 319 | 317 | 216 |
| mandala | 329 | err | 335 | 329 | 220 | 223 | 105 |

## graphene, e3nn eager: max |dH| / max |H| vs baseline (noise first)
| variant | noise | host | tf32 | safe_scatter | compile_stages | compile | cudagraphs | compile s (first pass, per stage) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| deeph | 8.4e-07 | 8.2e-07 | 4.5e-03 | 6.1e-07 | 1.7e+00 | 1.7e+00 | 1.7e+00 | compile_stages=9, compile=2, cudagraphs=19 |
| deeph_in_mandala | 6.4e-07 | 5.7e-07 | 3.7e-03 | 6.8e-07 | 9.6e-01 | 9.6e-01 | 9.6e-01 | compile_stages=4, compile=1, cudagraphs=16 |
| deeph_e3 | 6.8e-07 | 5.1e-07 | 6.2e-04 | 6.5e-07 | 6.5e-07 | 6.5e-07 | 8.1e-07 | compile_stages=164, compile=28, cudagraphs=314 |
| deeph_e3_in_mandala | 5.0e-07 | 5.0e-07 | 8.8e-04 | 6.4e-07 | 5.3e-07 | 5.5e-07 | 5.7e-07 | compile_stages=156, compile=33, cudagraphs=351 |
| mace_h | 6.8e-07 | 7.4e-07 | 6.5e-04 | 6.6e-07 | 7.4e-07 | 6.8e-07 | 7.9e-07 | compile_stages=109, compile=36, cudagraphs=398 |
| mace_h_in_mandala | 6.0e-07 | 6.0e-07 | 8.9e-04 | 6.9e-07 | 8.0e-07 | 8.2e-07 | 8.0e-07 | compile_stages=105, compile=42, cudagraphs=361 |
| mandala | 2.5e-07 | - | 3.4e-04 | 2.5e-07 | 2.5e-07 | 2.5e-07 | 2.5e-07 | compile_stages=214, compile=38, cudagraphs=1660 |
- deeph: host patches identical {'from_packed': True, 'to_blocks': True, 'bond_frames@operator_features': True, 'bond_neighborhood_relations': True}
- deeph_in_mandala: host patches identical {'from_packed': True, 'trace_alignment': True, 'to_blocks': True, 'bond_frames@operator_features': True, 'bond_neighborhood_relations': True}
- deeph_e3: host patches identical {'from_packed': True, 'to_blocks': True}
- deeph_e3_in_mandala: host patches identical {'from_packed': True, 'trace_alignment': True, 'to_blocks': True}
- mace_h: host patches identical {'from_packed': True, 'to_blocks': True}
- mace_h_in_mandala: host patches identical {'from_packed': True, 'trace_alignment': True, 'to_blocks': True}
- mandala: host patches identical {}; errors {'host': 'AttributeError("\'NoneType\' object has no attribute \'keys\'")'}

## mos2, e3nn eager: end_to_end median (ms)
| variant | baseline | host | tf32 | safe_scatter | compile_stages | compile | cudagraphs |
|---|---:|---:|---:|---:|---:|---:|---:|
| deeph | 3225 | 456 | 456 | 451 | 448 | 444 | 451 |
| deeph_in_mandala | 3330 | 480 | 456 | 469 | 468 | 474 | 482 |
| deeph_e3 | 980 | 530 | 524 | 516 | 539 | 516 | 480 |
| deeph_e3_in_mandala | 1222 | 528 | 512 | 489 | 535 | 537 | 512 |
| mace_h | 1145 | 669 | 723 | 677 | 562 | 545 | 455 |
| mace_h_in_mandala | 1550 | 694 | 672 | 702 | 554 | 559 | 485 |
| mandala | 905 | err | 940 | 917 | 859 | 806 | - |

## mos2, e3nn eager: end_to_end min (ms)
| variant | baseline | host | tf32 | safe_scatter | compile_stages | compile | cudagraphs |
|---|---:|---:|---:|---:|---:|---:|---:|
| deeph | 2938 | 440 | 433 | 439 | 431 | 432 | 422 |
| deeph_in_mandala | 3097 | 444 | 438 | 447 | 449 | 459 | 463 |
| deeph_e3 | 943 | 509 | 519 | 509 | 431 | 407 | 296 |
| deeph_e3_in_mandala | 1181 | 484 | 491 | 476 | 430 | 434 | 318 |
| mace_h | 1092 | 656 | 661 | 646 | 521 | 509 | 403 |
| mace_h_in_mandala | 1338 | 653 | 656 | 663 | 508 | 515 | 426 |
| mandala | 888 | err | 899 | 891 | 835 | 794 | - |

## mos2, e3nn eager: forward median (ms)
| variant | baseline | host | tf32 | safe_scatter | compile_stages | compile | cudagraphs |
|---|---:|---:|---:|---:|---:|---:|---:|
| deeph | 18 | 18 | 15 | 18 | 11 | 11 | 10 |
| deeph_in_mandala | 940 | 94 | 97 | 97 | 95 | 93 | 97 |
| deeph_e3 | 366 | 343 | 341 | 340 | 335 | 330 | 292 |
| deeph_e3_in_mandala | 1119 | 430 | 424 | 401 | 416 | 417 | 376 |
| mace_h | 483 | 498 | 486 | 567 | 368 | 357 | 280 |
| mace_h_in_mandala | 1291 | 588 | 581 | 585 | 454 | 462 | 383 |
| mandala | 492 | err | 495 | 495 | 414 | 391 | - |

## mos2, e3nn eager: forward min (ms)
| variant | baseline | host | tf32 | safe_scatter | compile_stages | compile | cudagraphs |
|---|---:|---:|---:|---:|---:|---:|---:|
| deeph | 18 | 18 | 15 | 18 | 11 | 11 | 10 |
| deeph_in_mandala | 739 | 93 | 94 | 94 | 92 | 89 | 91 |
| deeph_e3 | 336 | 317 | 337 | 335 | 242 | 235 | 116 |
| deeph_e3_in_mandala | 1088 | 423 | 403 | 390 | 325 | 325 | 203 |
| mace_h | 470 | 487 | 471 | 485 | 332 | 326 | 232 |
| mace_h_in_mandala | 1250 | 571 | 561 | 570 | 413 | 423 | 328 |
| mandala | 479 | err | 481 | 483 | 394 | 388 | - |

## mos2, e3nn eager: max |dH| / max |H| vs baseline (noise first)
| variant | noise | host | tf32 | safe_scatter | compile_stages | compile | cudagraphs | compile s (first pass, per stage) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| deeph | 5.2e-07 | 5.2e-07 | 4.6e-03 | 5.0e-07 | 1.0e+00 | 1.0e+00 | 1.0e+00 | compile_stages=19, compile=5, cudagraphs=9 |
| deeph_in_mandala | 5.5e-07 | 5.6e-07 | 4.4e-03 | 5.6e-07 | 1.0e+00 | 1.0e+00 | 1.0e+00 | compile_stages=10, compile=2, cudagraphs=5 |
| deeph_e3 | 4.0e-07 | 5.0e-07 | 7.0e-04 | 5.0e-07 | 4.7e-07 | 5.1e-07 | 5.1e-07 | compile_stages=283, compile=46, cudagraphs=208 |
| deeph_e3_in_mandala | 6.3e-07 | 7.8e-07 | 1.0e-03 | 7.6e-07 | 7.8e-07 | 8.4e-07 | 7.8e-07 | compile_stages=258, compile=59, cudagraphs=235 |
| mace_h | 3.4e-07 | 3.6e-07 | 5.9e-04 | 4.7e-07 | 5.1e-07 | 5.1e-07 | 4.7e-07 | compile_stages=234, compile=67, cudagraphs=168 |
| mace_h_in_mandala | 4.7e-07 | 4.5e-07 | 6.5e-04 | 6.3e-07 | 6.1e-07 | 5.8e-07 | 5.8e-07 | compile_stages=194, compile=72, cudagraphs=171 |
| mandala | 2.1e-07 | - | 4.8e-04 | 2.1e-07 | 2.1e-07 | 2.1e-07 | - | compile_stages=682, compile=128 |
- deeph: host patches identical {'from_packed': True, 'to_blocks': True, 'bond_frames@operator_features': True, 'bond_neighborhood_relations': True}
- deeph_in_mandala: host patches identical {'from_packed': True, 'trace_alignment': True, 'to_blocks': True, 'bond_frames@operator_features': True, 'bond_neighborhood_relations': True}
- deeph_e3: host patches identical {'from_packed': True, 'to_blocks': True}
- deeph_e3_in_mandala: host patches identical {'from_packed': True, 'trace_alignment': True, 'to_blocks': True}
- mace_h: host patches identical {'from_packed': True, 'to_blocks': True}
- mace_h_in_mandala: host patches identical {'from_packed': True, 'trace_alignment': True, 'to_blocks': True}
- mandala: host patches identical {}; errors {'host': 'AttributeError("\'NoneType\' object has no attribute \'keys\'")'}
