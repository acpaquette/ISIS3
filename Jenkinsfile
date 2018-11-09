pipeline {
    agent none
    environment {
        ISISROOT="${workspace}" + "/build/"
        ISIS3TESTDATA="/usgs/cpkgs/isis3/testData/"
        ISIS3DATA="/usgs/cpkgs/isis3/data/"
    }
    stages {
        stage('Fedora25') {
            agent {
                docker {
                    label 'docker'
                    image 'chrisryancombs/docker_isis'
                    args  '-v /scratch/krodriguez/isis3_data/data:/data -v /scratch/krodriguez/isis3_data/testData:/testData'
                }
            }
            steps {
                sh """
                    conda env create -n isis3 -f environment.yml
                    source activate isis3
                    mkdir -p ./install ./build && cd build
                    export ISISROOT=\$(pwd)
                    export ISIS3TESTDATA="/testData/"
                    export ISIS3DATA="/data/"
                    cmake -GNinja -DJP2KFLAG=OFF -Dpybindings=OFF -DCMAKE_INSTALL_PREFIX=../install -Disis3Data=/usgs/cpkgs/isis3/data -Disis3TestData=/usgs/cpkgs/isis3/testData ../isis
                    set +e
                    ninja -j8 && ninja install
                    ctest -V -R _unit_ --timeout 500
                    ctest -V -R _app_ --timeout 500
                    ctest -V -R _module_ --timeout 500
                    """
            }
        }
        // stage('CentOS7') {
        //     agent {
        //         docker {
        //             label 'docker'
        //             image 'chrisryancombs/docker_isis_centos'
        //             args  '''\
        //                      -v /scratch/krodriguez/isis3_data/data:/usgs/cpkgs/isis3/data \
        //                      -v /scratch/krodriguez/isis3_data/testData:/usgs/cpkgs/isis3/testData\
        //                      -v /usgs/cpkgs/isis3/isis3mgr_scripts:/usgs/cpkgs/isis3/isis3mgr_scripts
        //                   '''
        //         }
        //     }
        //     steps {
        //         sh """
        //             conda env create -n isis3 -f environment_gcc4.yml
        //             source activate isis3
        //             cd build
        //             set +e
        //             ./isis/scripts/isis3Startup.py
        //             ctest -R _unit_ --timeout 500
        //             ctest -R _app_ --timeout 500
        //             ctest -R _module_ --timeout 500
        //         """
        //     }
        // }
    }
}
//    post {
//        success {
//            sh 'pwd && ls'
//            archiveArtifacts artifacts: "build/objects/*.o"
//        }
//        always {
//            mail to: 'ccombs@usgs.gov',
//                    subject: "Build Finished: ${currentBuild.fullDisplayName}",
//                    body: "Link: ${env.BUILD_URL}"
//            sh "rm -rf build/* && rm -rf install/*"
//            cleanWs()
//        }
//    }
// }
