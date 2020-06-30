import Foundation

//KMeans 알고리즘 주석 달기
/*
 k-평균 알고리즘은 주어진 데이터를 k개의 클러스터로 묶는 알고리즘으로, 각 클러스터와 거리 차이의 분산을 최소화하는 방식으로 동작한다.
 */

class KMeans {
    let centroidCnt: Int //클러스터 중심(Centroid)의 갯수
    private(set) var centroids = [Vector]() //클러스터 중심(Centroid)들을 담은 리스트
    var clusterDict = [Vector:[Vector]]()

    
    //let kmm = KMeans(centroidCnt: 3) //3개의 centroid를 가지는 클러스터링을 하고자 한다. ->Kmeans의 객체를 생성
    init(centroidCnt: Int) {
        assert(centroidCnt > 1, "Exception: KMeans with less than 2 centers.") //Kmeans 알고리즘은 2개 이상의 중심을 가져야 정상적으로 작동한다.
        self.centroidCnt = centroidCnt
    }

    //해당 개체에 대해 어떤 centroid와 더 가까운지 판단해주는 함수
    private func indexOfNearestCenter(_ vecX: Vector, centers: [Vector]) -> Int {
        var nearestDist = Double.greatestFiniteMagnitude //가장 큰 수?를 넣어준다.
        var minIndex = 0

        for (idx, center) in centers.enumerated() {
            let dist = vecX.distanceTo(center) //해당 개체에서 centroid까지의 거리
            if dist < nearestDist {
                minIndex = idx
                nearestDist = dist
            }
        }
        return minIndex //가장 가까운 centroid의 index값 반환
    }

    //kmm.trainCenters(categoryCpyLocations, convergeDistance: 10)
    func trainCenters(_ points: [Vector], convergeDistance: Double) {
        
        let zeroVector = Vector([Double](repeating: 0, count: points[0].length))

        // Randomly take k objects from the input data to make the initial centroids.
        var centers = reservoirSample(points, end: centroidCnt)//클러스터링할 개체들과 군집 수를 매개변수로 담아 군집의 중심을 랜덤 초기화

        var centerMoveDist = 0.0
        
        repeat {
            //classification은 어떤 개체가 어떤 centroid에 속하는지를 알려준다.
            //This array keeps track of which data points belong to which centroids.
            var classification: [[Vector]] = .init(repeating: [], count: centroidCnt)

            
            //모든 개체에 대해서 어떤 centroid와 더 가까운지 탐색한다.
            // For each data point, find the centroid that it is closest to.
            for point in points {
                let classIndex = indexOfNearestCenter(point, centers: centers) //해당 개체에서 가장 가까운 centroid
                classification[classIndex].append(point)//해당 centroid에 개체가 속함을 알려준다.
            }

            // Take the average of all the data points that belong to each centroid.
            //해당 centroid에 속해있는 개체간의 중점?을 구한다.
            // This moves the centroid to a new position.
            //해당 centroid를 개체간의 중점으로 이동한다.
            let newCenters = classification.map { assignedPoints in
                assignedPoints.reduce(zeroVector, +) / Double(assignedPoints.count)
            }

            // Find out how far each centroid moved since the last iteration. If it's
            // only a small distance, then we're done.
            
            centerMoveDist = 0.0
            for idx in 0..<centroidCnt {
                centerMoveDist += centers[idx].distanceTo(newCenters[idx])
            }

            centers = newCenters
        } while centerMoveDist > convergeDistance

        centroids = centers
    }

    func fit(_ point: Vector) {
        assert(!centroids.isEmpty, "Exception: KMeans tried to fit on a non trained model.")

        let centroidIndex = indexOfNearestCenter(point, centers: centroids)
        let keyExists = clusterDict[centroids[centroidIndex]] != nil
        if keyExists {
            clusterDict[centroids[centroidIndex]]?.append(point)
        } else {
            clusterDict[centroids[centroidIndex]] = [point]
        }
    }

    func fit(_ points: [Vector]) -> [Vector:[Vector]] {
        assert(!centroids.isEmpty, "Exception: KMeans tried to fit on a non trained model.")

        points.forEach(fit)
        return clusterDict
    }

    // Pick k random elements from samples: 랜덤한 centroid를 추출한다.
    func reservoirSample<T>(_ samples: [T], end: Int) -> [T] {
        var result = [T]()

        // Fill the result array with first k elements
        for index in 0..<end {
            result.append(samples[index])
        }

        // Randomly replace elements from remaining pool
        for index in end..<samples.count {
            let nextIndex = Int(arc4random_uniform(UInt32(index + 1)))
            if nextIndex < end {
                result[nextIndex] = samples[index]
            }
        }
        return result
    }
}
