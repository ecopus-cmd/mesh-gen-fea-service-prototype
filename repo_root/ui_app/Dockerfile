FROM dolfinx/dolfinx:stable

# Remove broken gmsh shipped with dolfinx
RUN apt-get update && apt-get remove -y gmsh || true

# Install ALL runtime libraries required by the official Gmsh binary
RUN apt-get update && apt-get install -y \
    libxrender1 \
    libxext6 \
    libgl1 \
    libglu1-mesa \
    libfontconfig1 \
    libfreetype6 \
    libxkbcommon0 \
    libgomp1 \
    libgfortran5 \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Install Gmsh 4.11.1 (headless)
RUN wget https://gmsh.info/bin/Linux/gmsh-4.11.1-Linux64.tgz && \
    tar -xzf gmsh-4.11.1-Linux64.tgz && \
    mv gmsh-4.11.1-Linux64 /opt/gmsh && \
    rm gmsh-4.11.1-Linux64.tgz

# Make gmsh globally available
RUN ln -sf /opt/gmsh/bin/gmsh /usr/local/bin/gmsh

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
COPY . .

CMD ["python3", "app.py"]
