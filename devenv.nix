{ pkgs, lib, ... }:

let
  texEnvironment = pkgs.texlive.combine {
    inherit (pkgs.texlive) preview scheme-medium standalone;
  };
  runtimeLibraries = with pkgs; [
    alsa-lib
    at-spi2-atk
    at-spi2-core
    brotli
    cairo
    cups
    dbus
    expat
    fontconfig
    freetype
    glib
    libGL
    libgbm
    libdrm
    libxkbcommon
    krb5
    mesa
    nspr
    nss
    stdenv.cc.cc
    systemd
    vulkan-loader
    wayland
    xorg.libICE
    xorg.libSM
    xorg.libX11
    xorg.libXcomposite
    xorg.libXcursor
    xorg.libXdamage
    xorg.libXext
    xorg.libXfixes
    xorg.libXi
    xorg.libXrandr
    xorg.libXrender
    xorg.libXtst
    xorg.libxcb
    xorg.libxkbfile
    zlib
    zstd
  ];
in
{
  packages = with pkgs; [
    ghostscript
    git
    grim
    imagemagick
    nodejs
    pkg-config
    slurp
    texEnvironment
    uv
    wl-clipboard
  ];

  languages.python = {
    enable = true;
    package = pkgs.python311;
    uv.enable = true;
  };

  env = {
    NO_ALBUMENTATIONS_UPDATE = "1";
    PYTHONNOUSERSITE = "1";
    QT_STYLE_OVERRIDE = "Fusion";
    UV_PYTHON_DOWNLOADS = "never";
    UV_PYTHON_PREFERENCE = "only-system";
  };

  enterShell = ''
    export LD_LIBRARY_PATH="${lib.makeLibraryPath runtimeLibraries}:/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT"
    export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
    echo "LaTeX-OCR devenv"
    python --version
    uv sync --all-extras --group dev --frozen 2>/dev/null \
      || uv sync --all-extras --group dev \
      || exit $?
  '';

  enterTest = ''
    uv run pytest
  '';
}
