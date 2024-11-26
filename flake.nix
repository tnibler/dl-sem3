{
  description = "Python 3.9 development environment";

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config.allowUnfree = true;
    };
    vsc =
      pkgs.vscode-with-extensions.override
      {
        vscodeExtensions = with pkgs.vscode-extensions; [
          ms-toolsai.jupyter
          ms-toolsai.vscode-jupyter-cell-tags
          vscodevim.vim
          ms-python.python
          ms-python.vscode-pylance
        ];
      };
  in {
    devShells.${system}.default = pkgs.mkShell {
      nativeBuildInputs = with pkgs;
        [
          python312
          stdenv.cc.cc.lib
        ]
        ++ (with python312Packages; [
          pip
          numpy
          pytorch
          torchvision
          virtualenv
          basedpyright
          python-lsp-server
          python-lsp-black
          python-lsp-server
          python-lsp-jsonrpc
          python-lsp-black
          python-lsp-ruff
          pyls-isort
          pyls-flake8
          flake8
          isort
          black
          jupyter
          matplotlib
        ])
        ++ [
          vsc
        ];
      LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [pkgs.stdenv.cc.cc];
    };
  };
}
