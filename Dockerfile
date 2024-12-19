FROM ubuntu:22.04

RUN apt-get update -y
# for tdata geo locn
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

# Install dependencies
RUN apt-get -y install python3.11 python3-pip git binutils-multiarch wget
# Ghidra
RUN apt-get -y install  openjdk-17-jdk openjdk-11-jdk

# Joern
RUN apt-get install -y openjdk-11-jre-headless psmisc && \
     apt-get clean;
RUN pip3 install cpgqls_client regex

WORKDIR /varbert_workdir

# Dwarfwrite
RUN git clone https://github.com/rhelmot/dwarfwrite.git
WORKDIR /varbert_workdir/dwarfwrite
RUN pip install .

WORKDIR /varbert_workdir/
RUN git clone https://github.com/sefcom/VarBERT.git
WORKDIR /varbert_workdir/VarBERT/
RUN pip install -r requirements.txt

RUN apt-get install unzip
RUN wget -cO /varbert_workdir/joern.tar.gz "https://www.dropbox.com/scl/fi/toh6087y5t5xyln47i5ih/modified_joern.tar.gz?rlkey=lfvjn1u7zvtp9a4cu8z8vgsof" && tar -xzf /varbert_workdir/joern.tar.gz -C /varbert_workdir && rm /varbert_workdir/joern.tar.gz
RUN wget -cO /varbert_workdir/ghidra_10.4_PUBLIC_20230928.zip "https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_10.4_build/ghidra_10.4_PUBLIC_20230928.zip" && unzip /varbert_workdir/ghidra_10.4_PUBLIC_20230928.zip -d /varbert_workdir && rm /varbert_workdir/ghidra_10.4_PUBLIC_20230928.zip

