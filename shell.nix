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
in python38.withPackages (ps: with ps; [
  numpy
  scipy
  meshio
  matplotlib
  ipython
  skfem310
])).env
