with import <nixpkgs> {};

(let
  skfem310 = python38.pkgs.buildPythonPackage rec {
    pname = "scikit-fem";
    version = "3.1.0";
    format = "pyproject";
    propagatedBuildInputs = [
      python38.pkgs.numpy
      python38.pkgs.scipy
      python38.pkgs.meshio
    ];
    src = python38.pkgs.fetchPypi {
      inherit pname version;
      sha256 = "017nk05qrd6phwyfxyb1zybcyz579l9rgvsh37bcwxajiy1n2k6n";
    };
    doCheck = false;
  };
  skfem520 = python38.pkgs.buildPythonPackage rec {
    pname = "scikit-fem";
    version = "5.2.0";
    format = "pyproject";
    propagatedBuildInputs = [
      python38.pkgs.numpy
      python38.pkgs.scipy
    ];
    src = python38.pkgs.fetchPypi {
      inherit pname version;
      sha256 = "0yf8pz4ynzzmsypqdmp8xyfj2al5az6qr2skdfzvjq1pyqwqi5jz";
    };
    doCheck = false;
  };
in python38.withPackages (ps: with ps; [
  numpy
  scipy
  meshio
  matplotlib
  ipython
  skfem520
])).env
