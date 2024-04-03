{
  description = "HVM-Core";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.fenix.url = "github:nix-community/fenix/monthly";

  outputs = { self, nixpkgs, flake-utils, fenix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        toolchain = fenix.packages.${system}.fromToolchainFile {
          file = ./rust-toolchain.toml;
          # to recompute the hash, replace with `pkgs.lib.fakeSha256`
          sha256 = "sha256-hBzihtLpwbCyL5AgwKj0sUbJXRir18utWgqUZHGsbFs=";
        };
      in {
        devShells.default =
          pkgs.mkShell { packages = [ toolchain pkgs.clippy ]; };
      });
}
