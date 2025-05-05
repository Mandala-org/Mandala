import pytest
from pathlib import Path

import torch

from core.orbital_irrep_config import OrbitalIrrepConfig
from data.openmx_parser import parse_openmx_scfout


@pytest.fixture(scope="module")
def orbital_cfg():
    # H: 2s + 1p (dim 5);  O: 3s + 2p (dim 7)
    return OrbitalIrrepConfig.from_dict(
        {
            "H": ["2x0e", "1x1o"],
            "O": ["3x0e", "2x1o"],
        }
    )


def test_parse(orbital_cfg: OrbitalIrrepConfig):
    sample = Path("./data/small/H2O/H2O_original.out")
    atoms = list("HHHHOO")  # global order

    mats = parse_openmx_scfout(sample, atoms, orbital_cfg)
    assert "hamiltonian" in mats and "overlap" in mats and "density" in mats

    hamiltonian = mats["hamiltonian"]
    overlap = mats["overlap"]
    density = mats["density"]
    # there should be at least one H‑O block
    assert "H-O" in hamiltonian.keys()
    E_HO, d_H, d_O = hamiltonian["H-O"].shape
    assert d_H == 5 and d_O == 9

    # round‑trip vector check
    snap_vec = hamiltonian.to_vectors()
    assert snap_vec["H-O"].shape[-1] == hamiltonian.mapper.vector_dim("H-O")

    assert hamiltonian[0, 0].shape == (5, 5)
    assert hamiltonian[1, 4].shape == (5, 9)
    assert hamiltonian[5, 3].shape == (9, 5)
    assert hamiltonian[4, 5].shape == (9, 9)

    assert overlap[1, 3].shape == (5, 5)
    assert overlap[2, 5].shape == (5, 9)
    assert overlap[4, 0].shape == (9, 5)
    assert overlap[5, 4].shape == (9, 9)

    assert density[2, 3].shape == (5, 5)
    assert density[3, 4].shape == (5, 9)
    assert density[4, 3].shape == (9, 5)
    assert density[5, 5].shape == (9, 9)


def test_value(orbital_cfg: OrbitalIrrepConfig):
    sample = Path("./data/small/H2O/H2O_original.out")
    atoms = list("HHHHOO")

    mats = parse_openmx_scfout(sample, atoms, orbital_cfg)
    hamiltonian = mats["hamiltonian"]
    overlap = mats["overlap"]
    density = mats["density"]

    hamiltonian_4_2 = torch.tensor(
        [
            [
                -0.0041153241786031,
                0.0104082237482741,
                0.0104477406877440,
                0.0012862654628771,
                0.0099462777440482,
            ],
            [
                -0.0084789314184493,
                -0.0023620026873997,
                -0.0019589400865788,
                -0.0107783218239291,
                -0.0010673308872907,
            ],
            [
                -0.0356825466116493,
                0.0362212361888985,
                0.0402489724168223,
                -0.0193380264010976,
                0.0401130689637325,
            ],
            [
                -0.0062891217056391,
                0.0119772907792404,
                0.0112371824386699,
                0.0002782206657456,
                0.0136116849614407,
            ],
            [
                -0.0022600546749644,
                0.0049562717042971,
                0.0053658863438921,
                -0.0037492224362208,
                0.0051894175204805,
            ],
            [
                -0.0058837498418362,
                0.0111497471869116,
                0.0132677981630397,
                0.0003035684667232,
                0.0097590125145755,
            ],
            [
                0.0174609894000005,
                -0.0237632678498975,
                -0.0233863273206632,
                0.0044959238226331,
                -0.0305115046144773,
            ],
            [
                0.0076907011228044,
                -0.0154813697190676,
                -0.0170551868240803,
                0.0125902843144436,
                -0.0165569721892516,
            ],
            [
                0.0162541406716508,
                -0.0217493675371054,
                -0.0291541500486089,
                0.0041195767590602,
                -0.0200725586723728,
            ],
        ]
    )
    assert torch.allclose(
        hamiltonian[4, 2], hamiltonian_4_2, atol=1e-9
    ), "Hamiltonian value mismatch"

    overlap_5_5 = torch.tensor(
        [
            [
                0.9999999976585413,
                0.0000000615537784,
                -0.0000001543650183,
                -0.0000000000604156,
                0.0000000000000412,
                -0.0000000000000302,
                0.0000000000050411,
                0.0000000000000268,
                -0.0000000000000222,
            ],
            [
                0.0000000615537943,
                0.9999990177650994,
                -0.0000016072424590,
                -0.0000000000311108,
                0.0000000000000013,
                0.0000000000000452,
                -0.0000000000496809,
                0.0000000000000136,
                0.0000000000000217,
            ],
            [
                -0.0000001543650825,
                -0.0000016072424613,
                0.9999884651156814,
                0.0000000000241664,
                -0.0000000000000363,
                0.0000000000000304,
                0.0000000000003127,
                -0.0000000000000099,
                -0.0000000000000370,
            ],
            [
                0.0000000000604063,
                0.0000000000310986,
                -0.0000000000242125,
                0.9999999596205204,
                0.0000000000000473,
                -0.0000000000000207,
                0.0000001115758861,
                0.0000000000000308,
                0.0000000000000419,
            ],
            [
                0.0000000000000499,
                0.0000000000000339,
                0.0000000000000138,
                0.0000000000000271,
                0.9999999596205237,
                0.0000000000000270,
                -0.0000000000000430,
                0.0000001115759457,
                0.0000000000000026,
            ],
            [
                -0.0000000000000282,
                0.0000000000000113,
                0.0000000000000024,
                -0.0000000000000100,
                0.0000000000000392,
                0.9999999596204995,
                -0.0000000000000414,
                -0.0000000000000308,
                0.0000001115759171,
            ],
            [
                -0.0000000000050186,
                0.0000000000496745,
                -0.0000000000003732,
                0.0000001115759358,
                -0.0000000000000233,
                0.0000000000000040,
                1.0000004255551198,
                0.0000000000000432,
                0.0000000000000431,
            ],
            [
                -0.0000000000000151,
                -0.0000000000000042,
                0.0000000000000471,
                -0.0000000000000125,
                0.0000001115759268,
                0.0000000000000013,
                0.0000000000000221,
                1.0000004255551045,
                0.0000000000000239,
            ],
            [
                -0.0000000000000436,
                -0.0000000000000437,
                0.0000000000000402,
                0.0000000000000168,
                0.0000000000000032,
                0.0000001115758547,
                0.0000000000000140,
                -0.0000000000000146,
                1.0000004255551451,
            ],
        ]
    )
    assert torch.allclose(
        overlap[5, 5], overlap_5_5, atol=1e-9
    ), "Overlap value mismatch"

    density_2_2 = torch.tensor(
        [
            [
                0.4376985836393557,
                -0.0094285564292734,
                0.0042281100759622,
                0.0006137711777119,
                -0.0033890620306084,
            ],
            [
                -0.0094285564292734,
                0.0006956521916124,
                -0.0005087069896141,
                0.0004431693464505,
                -0.0003065522122476,
            ],
            [
                0.0042281100759622,
                -0.0005087069896141,
                0.0007144814157878,
                -0.0005951557380101,
                0.0003705061743471,
            ],
            [
                0.0006137711777119,
                0.0004431693464505,
                -0.0005951557380101,
                0.0006451203794178,
                -0.0003963128627393,
            ],
            [
                -0.0033890620306084,
                -0.0003065522122476,
                0.0003705061743471,
                -0.0003963128627393,
                0.0003803518935681,
            ],
        ]
    )
    assert torch.allclose(
        density[2, 2], density_2_2, atol=1e-9
    ), "Density value mismatch"


def test_value_pbc(orbital_cfg: OrbitalIrrepConfig):
    sample = Path("./data/small/H2O/H2O_pbc_original.out")
    atoms = list("HHHHOO")

    mats = parse_openmx_scfout(sample, atoms, orbital_cfg)
    hamiltonian = mats["hamiltonian"]
    overlap = mats["overlap"]
    density = mats["density"]

    hamiltonian_3_4 = torch.tensor(
        [
            [
                -0.7092862129,
                -0.2322716713,
                -0.9319261312,
                -0.1521144956,
                0.1521169990,
                0.2043066323,
                -0.0240604728,
                0.0240555461,
                0.0455909669,
            ],
            [
                0.3793754280,
                0.2470810115,
                1.0171759129,
                -0.1870319843,
                0.1870281845,
                0.1366036832,
                0.0082880706,
                -0.0082799299,
                -0.0103982277,
            ],
            [
                0.2551721334,
                0.0789289474,
                -0.0429678708,
                -0.0967083946,
                -0.0180612355,
                -0.0677458644,
                -0.0554984063,
                -0.0166817307,
                -0.0111170970,
            ],
            [
                -0.2551709116,
                -0.0789319575,
                0.0429752693,
                -0.0180592537,
                -0.0967120081,
                0.0677454025,
                -0.0166822411,
                -0.0554961562,
                0.0111178420,
            ],
            [
                -0.2254575640,
                -0.0948449522,
                0.0272943005,
                -0.0306348260,
                0.0306365546,
                -0.0496056899,
                -0.0144353975,
                0.0144337695,
                -0.0453140736,
            ],
        ]
    )
    assert torch.allclose(
        hamiltonian[3, 4], hamiltonian_3_4, atol=1e-8
    ), "Hamiltonian value mismatch"

    overlap_4_5 = torch.tensor(
        [
            [
                7.6841272414e-02,
                3.4702867270e-01,
                1.2430062294e00,
                -4.3364707381e-09,
                -3.9817678044e-09,
                -3.0559021980e-10,
                3.4342519939e-09,
                1.1223164620e-09,
                6.4210325945e-10,
            ],
            [
                3.4702867270e-01,
                -6.2971793115e-02,
                -4.1830760241e-01,
                -6.9849193096e-09,
                1.1932570487e-09,
                -1.0186340660e-09,
                -1.5832483768e-08,
                1.1292286217e-08,
                -5.8207660913e-09,
            ],
            [
                1.2430062294e00,
                -4.1830760241e-01,
                2.9495179653e00,
                -1.3969838619e-08,
                -1.9790604711e-09,
                5.2968971431e-09,
                1.8626451492e-08,
                2.2584572434e-08,
                -6.0535967350e-09,
            ],
            [
                4.3364707381e-09,
                6.9849193096e-09,
                1.3969838619e-08,
                -9.1970436275e-02,
                2.6193447411e-10,
                2.0736479200e-10,
                2.7188524604e-01,
                1.1874362826e-08,
                -2.7212081477e-09,
            ],
            [
                3.9817678044e-09,
                -1.1932570487e-09,
                1.9790604711e-09,
                2.6193447411e-10,
                -9.1970458627e-02,
                -6.9121597335e-11,
                1.1874362826e-08,
                2.7188521624e-01,
                -1.1510564946e-08,
            ],
            [
                3.0559021980e-10,
                1.0186340660e-09,
                -5.2968971431e-09,
                2.0736479200e-10,
                -6.9121597335e-11,
                -8.0344036222e-02,
                -2.7212081477e-09,
                -1.1510564946e-08,
                2.3560644686e-01,
            ],
            [
                -3.4342519939e-09,
                1.5832483768e-08,
                -1.8626451492e-08,
                2.7188524604e-01,
                1.1874362826e-08,
                -2.7212081477e-09,
                -3.2656052709e-01,
                2.3166649044e-08,
                4.4819898903e-09,
            ],
            [
                -1.1223164620e-09,
                0.0000000000e00,
                0.0000000000e00,
                1.4901161194e-08,
                2.7188524604e-01,
                0.0000000000e00,
                1.4901161194e-08,
                -3.2656049728e-01,
                0.0000000000e00,
            ],
            [
                1.4901161194e-08,
                0.0000000000e00,
                0.0000000000e00,
                7.4505805969e-09,
                0.0000000000e00,
                2.3560643196e-01,
                -1.4901161194e-08,
                0.0000000000e00,
                -2.5250178576e-01,
            ],
        ]
    )
    assert torch.allclose(
        overlap[4, 5], overlap_4_5, atol=1e-7
    ), "Overlap value mismatch"

    density_2_3 = torch.tensor(
        [
            [-5.4044947624, -0.1847825795, 0.6296055317, -0.6291579008, -1.6667642593],
            [-0.1847826242, 0.1810163110, -0.0788071081, 0.0787528008, -0.1885431260],
            [-0.6291579604, 0.0787527785, 0.4712260962, 0.4638453424, -0.3373683989],
            [0.6296054721, -0.0788071156, 0.4635453522, 0.4712260067, 0.3374770880],
            [-1.6667642593, -0.1885431260, 0.3374771774, -0.3373682797, -0.3775940537],
        ]
    )
    assert torch.allclose(
        density[2, 3], density_2_3, atol=1e-8
    ), "Density value mismatch"


def test_parse_pbc_shapes(orbital_cfg: OrbitalIrrepConfig):
    sample = Path("./data/small/H2O/H2O_pbc_original.out")
    atoms = list("HHHHOO")

    mats = parse_openmx_scfout(sample, atoms, orbital_cfg, pbc_sum=True)
    density = mats["density"]
    # ensure that duplicate Rn blocks were summed: count of H‑H edges is 16 (fully connected dir graph)
    assert density["H-H"].shape == (16, 5, 5)
    assert density["O-O"].shape == (4, 9, 9)
    assert density["H-O"].shape == (8, 5, 9)
    assert density["O-H"].shape == (8, 9, 5)
    # assert O-H is the same as H-O.T
    density = density.standardize_edges()
    density_T = density.transpose().standardize_edges()
    assert torch.allclose(
        density["O-H"], density_T["O-H"], atol=1e-5
    ), "D[O-H] should be the same as D[H-O].T"

    hamiltonian = mats["hamiltonian"]
    assert hamiltonian["H-H"].shape == (16, 5, 5)
    assert hamiltonian["O-O"].shape == (4, 9, 9)
    assert hamiltonian["H-O"].shape == (8, 5, 9)
    assert hamiltonian["O-H"].shape == (8, 9, 5)
    # assert O-H is the same as H-O.T
    hamiltonian = hamiltonian.standardize_edges()
    hamiltonian_T = hamiltonian.transpose().standardize_edges()
    assert torch.allclose(
        hamiltonian["O-H"], hamiltonian_T["O-H"], atol=1e-5
    ), "Ham[O-H] should be the same as Ham[H-O].T"

    overlap = mats["overlap"]
    assert overlap["H-H"].shape == (16, 5, 5)
    assert overlap["O-O"].shape == (4, 9, 9)
    assert overlap["H-O"].shape == (8, 5, 9)
    assert overlap["O-H"].shape == (8, 9, 5)
    # assert O-H is the same as H-O.T
    overlap = overlap.standardize_edges()
    overlap_T = overlap.transpose().standardize_edges()
    assert torch.allclose(
        overlap["O-H"], overlap_T["O-H"], atol=1e-5
    ), "Overlap[O-H] should be the same as Overlap[H-O].T"
