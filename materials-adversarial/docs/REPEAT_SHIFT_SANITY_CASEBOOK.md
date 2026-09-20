# Repeat/shift sanity casebook

All eight pairs pass the strict periodic validator, exact tokenizer round-trip,
zero-unknown, and no-truncation gates. `SEGMENTATION` means a doubled repeat cell
followed by a boundary translation; `SHIFT` means a one-cell boundary translation.
Tokens are separated by spaces below.

## Five high-drift cases

### poly_8 / poly_8_rs006 — SEGMENTATION

- Equivalence: `repeated_cell_periodic_graph_match`
- Source: `[*]C#CC(=C([*])c1cnc2ccccc2c1)c1cnc2ccccc2c1`
- Variant: `C(C#CC(=C(C#CC(c1cnc2ccccc2c1)=[*])c1cnc2ccccc2c1)c1cnc2ccccc2c1)(c1cnc2ccccc2c1)=[*]`
- Source tokens: `[*] C # C C ( = C ( [*] ) c 1 c n c 2 c c c c c 2 c 1 ) c 1 c n c 2 c c c c c 2 c 1`
- Variant tokens: `C ( C # C C ( = C ( C # C C ( c 1 c n c 2 c c c c c 2 c 1 ) = [*] ) c 1 c n c 2 c c c c c 2 c 1 ) c 1 c n c 2 c c c c c 2 c 1 ) ( c 1 c n c 2 c c c c c 2 c 1 ) = [*]`
- Transformer: `0.794209 -> 3.653364 eV`; drift `2.859155 eV`
- GraphMPNN: `2.973276 -> 2.717172 eV`; drift `0.256104 eV`

### poly_954 / poly_954_rs007 — SHIFT

- Equivalence: `periodic_star_linked_graph_match`
- Source: `[*]Oc1ccc(C(C)(C)c2ccc(OC(=O)C=CC([*])=O)cc2)cc1`
- Variant: `C(=CC(Oc1ccc(C(c2ccc(OC(=O)[*])cc2)(C)C)cc1)=O)[*]`
- Source tokens: `[*] O c 1 c c c ( C ( C ) ( C ) c 2 c c c ( O C ( = O ) C = C C ( [*] ) = O ) c c 2 ) c c 1`
- Variant tokens: `C ( = C C ( O c 1 c c c ( C ( c 2 c c c ( O C ( = O ) [*] ) c c 2 ) ( C ) C ) c c 1 ) = O ) [*]`
- Transformer: `5.008094 -> 2.373867 eV`; drift `2.634227 eV`
- GraphMPNN: `4.059274 -> 3.894649 eV`; drift `0.164625 eV`

### poly_8 / poly_8_rs007 — SEGMENTATION

- Equivalence: `repeated_cell_periodic_graph_match`
- Source: `[*]C#CC(=C([*])c1cnc2ccccc2c1)c1cnc2ccccc2c1`
- Variant: `C(=C(C#CC(=C(C#C[*])c1cnc2ccccc2c1)c1cnc2ccccc2c1)c1cnc2ccccc2c1)(c1cnc2ccccc2c1)[*]`
- Source tokens: `[*] C # C C ( = C ( [*] ) c 1 c n c 2 c c c c c 2 c 1 ) c 1 c n c 2 c c c c c 2 c 1`
- Variant tokens: `C ( = C ( C # C C ( = C ( C # C [*] ) c 1 c n c 2 c c c c c 2 c 1 ) c 1 c n c 2 c c c c c 2 c 1 ) c 1 c n c 2 c c c c c 2 c 1 ) ( c 1 c n c 2 c c c c c 2 c 1 ) [*]`
- Transformer: `0.794209 -> 3.341536 eV`; drift `2.547326 eV`
- GraphMPNN: `2.973276 -> 2.777951 eV`; drift `0.195325 eV`

### poly_443 / poly_443_rs001 — SEGMENTATION

- Equivalence: `repeated_cell_periodic_graph_match`
- Source: `[*]c1[nH]c([*])c2c1CCCC2`
- Variant: `c1([*])[nH]c(-c2[nH]c([*])c3c2CCCC3)c2c1CCCC2`
- Source tokens: `[*] c 1 [nH] c ( [*] ) c 2 c 1 C C C C 2`
- Variant tokens: `c 1 ( [*] ) [nH] c ( - c 2 [nH] c ( [*] ) c 3 c 2 C C C C 3 ) c 2 c 1 C C C C 2`
- Transformer: `1.834188 -> 4.273354 eV`; drift `2.439166 eV`
- GraphMPNN: `3.598621 -> 3.087885 eV`; drift `0.510736 eV`

### poly_29 / poly_29_rs002 — SEGMENTATION

- Equivalence: `repeated_cell_periodic_graph_match`
- Source: `[*]C=C([*])c1cccc(C#C)n1`
- Variant: `C(C(=CC(c1cccc(C#C)n1)=[*])c1cccc(C#C)n1)=[*]`
- Source tokens: `[*] C = C ( [*] ) c 1 c c c c ( C # C ) n 1`
- Variant tokens: `C ( C ( = C C ( c 1 c c c c ( C # C ) n 1 ) = [*] ) c 1 c c c c ( C # C ) n 1 ) = [*]`
- Transformer: `0.855260 -> 3.193443 eV`; drift `2.338183 eV`
- GraphMPNN: `2.771085 -> 2.477921 eV`; drift `0.293165 eV`

## Three low-drift cases

### poly_1595 / poly_1595_rs004 — SHIFT

- Equivalence: `periodic_star_linked_graph_match`
- Source / variant: `[*]CCCCCCCCCOC(=O)c1ccc(-c2ccc(C(=O)O[*])cc2)cc1` / `C(CCCCOC(c1ccc(-c2ccc(C(OCCCC[*])=O)cc2)cc1)=O)[*]`
- Token change: `[*] C C C C C C C C C ... O [*]` becomes `C ( C C C C O C ... O C C C C [*] ... ) [*]` (full strings above round-trip exactly).
- Transformer: `4.276936 -> 4.278684 eV`; drift `0.001748 eV`
- GraphMPNN drift: `0.033744 eV`

### poly_378 / poly_378_rs012 — SHIFT

- Equivalence: `periodic_star_linked_graph_match`
- Source / variant: the source starts `[*]c1ccc2...C([*])(F)F`; the variant starts `C(C(C(C(c1ccc2...` and ends `...(F)(F)[*]`. The exact strings and token arrays are retained in `variants.csv`.
- Token change: the two `[*]` tokens move from the ends of the aromatic/fluoroalkyl repeat-cell path to a translated cell boundary; all 123 source and 127 variant tokens remain known.
- Transformer: `3.020979 -> 3.018466 eV`; drift `0.002513 eV`
- GraphMPNN drift: `0.156291 eV`

### poly_567 / poly_567_rs002 — SEGMENTATION

- Equivalence: `repeated_cell_periodic_graph_match`
- Source: `[*]C(=O)OC(S1)=CC=C1C(=S)[*]`
- Variant: `O(C(=O)C(c1sc(OC(C(c2sc([*])cc2)=S)=O)cc1)=S)[*]`
- Source tokens: `[*] C ( = O ) O C ( S 1 ) = C C = C 1 C ( = S ) [*]`
- Variant tokens: `O ( C ( = O ) C ( c 1 s c ( O C ( C ( c 2 s c ( [*] ) c c 2 ) = S ) = O ) c c 1 ) = S ) [*]`
- Transformer: `3.037786 -> 3.040965 eV`; drift `0.003180 eV`
- GraphMPNN drift: `0.667762 eV`

The last example is a useful warning: the legacy GraphMPNN consumes the finite
dummy-atom graph, not the periodic quotient graph, so it is not invariant to all
genuine repeat/shift transformations.
