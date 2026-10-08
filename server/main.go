package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"os"
	"path/filepath"
	"sync"
	"time"

	pb "github.com/VincentSamuelPaul/distributedneuralnet/server/proto"
	"google.golang.org/grpc"
)

type server struct {
	pb.UnimplementedParameterServerServer
	gradients          [][]float32
	receivedCount      int
	mtx                sync.Mutex
	cond               *sync.Cond
	averagedIndices    []int32
	averagedValues     []float32
	totalSize          int32
	initialWeights     []float32
	weightsInitialized bool
	nWorkers           int
	round              int
	CheckpointInterval int
	CheckpointPath     string
	weightCount        int
	receivedWeights    [][]float32
	weightCond         *sync.Cond
	modelWeights       []float32
}

type Checkpoint struct {
	Round   int       `json:"round"`
	Weights []float32 `json:"weights"`
}

func (s *server) SendGradients(ctx context.Context, req *pb.GradientRequest) (*pb.WeightResponse, error) {
	s.mtx.Lock()
	defer s.mtx.Unlock()

	fullGrad := make([]float32, req.TotalSize)
	for i, idx := range req.Indices {
		fullGrad[idx] = req.Values[i]
	}

	s.gradients = append(s.gradients, fullGrad)
	s.receivedCount++

	totalSize := req.TotalSize
	currentRound := s.round
	if s.receivedCount == 1 {
		go func() {
			time.Sleep(60 * time.Second)
			s.mtx.Lock()
			if s.round == currentRound && s.receivedCount > 0 && s.receivedCount < s.nWorkers {
				log.Printf("Timeout: proceeding with %d/%d workers", s.receivedCount, s.nWorkers)
				s.nWorkers = s.receivedCount
				s.computeAverage(totalSize)
				s.cond.Broadcast()
			}
			s.mtx.Unlock()
		}()
	}

	if s.receivedCount < s.nWorkers {
		s.cond.Wait()
	} else {
		s.computeAverage(totalSize)
		s.cond.Broadcast()
	}

	return &pb.WeightResponse{
		Indices:   s.averagedIndices,
		Values:    s.averagedValues,
		TotalSize: s.totalSize,
	}, nil
}

func (s *server) SendWeights(ctx context.Context, req *pb.WeightRequest) (*pb.WeightResponse, error) {
	s.mtx.Lock()
	defer s.mtx.Unlock()

	s.receivedWeights = append(s.receivedWeights, req.Weights)
	s.weightCount++

	if s.weightCount < s.nWorkers {
		s.weightCond.Wait()
	} else {
		averaged := make([]float32, len(s.receivedWeights[0]))
		for i := range averaged {
			sum := float32(0)
			for _, w := range s.receivedWeights {
				sum += w[i]
			}
			averaged[i] = sum / float32(len(s.receivedWeights))
		}

		s.modelWeights = averaged
		s.weightCount = 0
		s.receivedWeights = nil

		saveCheckpoint(s.CheckpointPath, s.modelWeights, s.round)
		s.weightCond.Broadcast()
	}

	indices := make([]int32, len(s.modelWeights))
	for i := range indices {
		indices[i] = int32(i)
	}

	return &pb.WeightResponse{
		Indices:   indices,
		Values:    s.modelWeights,
		TotalSize: int32(len(s.modelWeights)),
	}, nil
}

func (s *server) computeAverage(totalSize int32) {
	averaged := make([]float32, totalSize)
	for i := range averaged {
		sum := float32(0)
		for _, workerGrads := range s.gradients {
			sum += workerGrads[i]
		}
		averaged[i] = sum / float32(len(s.gradients))
	}

	indices := make([]int32, len(averaged))
	for i := range indices {
		indices[i] = int32(i)
	}
	s.averagedIndices = indices
	s.averagedValues = averaged
	s.totalSize = totalSize
	s.receivedCount = 0
	s.gradients = nil
	s.round++
}

func saveCheckpoint(path string, values []float32, round int) {
	checkpoint := Checkpoint{Round: round, Weights: values}

	data, err := json.Marshal(checkpoint)
	if err != nil {
		log.Printf("Error marshalling checkpoint: %v", err)
		return
	}

	err = os.WriteFile(path, data, 0644)
	if err != nil {
		log.Printf("Error writing checkpoint: %v", err)
		return
	}

	log.Printf("Checkpoint saved at round %d", round)
}

func loadCheckpoint(path string) *Checkpoint {
	var checkpoint Checkpoint

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		log.Printf("Error reading checkpoint: %v", err)
		return nil
	}

	err = json.Unmarshal(data, &checkpoint)
	if err != nil {
		log.Printf("Error unmarshalling data from json file: %v", err)
		return nil
	}

	log.Printf("Loaded checkpoint from round %d", checkpoint.Round)
	return &checkpoint
}

func (s *server) InitializeWeights(ctx context.Context, req *pb.WeightInit) (*pb.WeightResponse, error) {
	s.mtx.Lock()
	defer s.mtx.Unlock()

	if len(s.modelWeights) > 0 {
		s.initialWeights = s.modelWeights
		log.Printf("Using checkpoint weights for initialization")
	} else {
		s.initialWeights = req.Values
	}

	s.weightsInitialized = true
	indices := make([]int32, len(s.initialWeights))
	for i := range indices {
		indices[i] = int32(i)
	}
	return &pb.WeightResponse{
		Indices:   indices,
		Values:    s.initialWeights,
		TotalSize: int32(len(s.initialWeights)),
	}, nil
}

func (s *server) GetInitialWeights(ctx context.Context, req *pb.Empty) (*pb.WeightResponse, error) {
	s.mtx.Lock()
	defer s.mtx.Unlock()
	indices := make([]int32, len(s.initialWeights))
	for i := range indices {
		indices[i] = int32(i)
	}
	return &pb.WeightResponse{
		Indices:   indices,
		Values:    s.initialWeights,
		TotalSize: int32(len(s.initialWeights)),
	}, nil
}

func main() {
	cfg := loadConfig()
	os.MkdirAll(filepath.Dir(cfg.Server.CheckpointPath), 0755)
	checkpoint := loadCheckpoint(cfg.Server.CheckpointPath)

	PORT := fmt.Sprintf(":%d", cfg.Server.Port)
	listen, err := net.Listen("tcp", PORT)
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}
	s := grpc.NewServer()
	srv := &server{}
	if checkpoint != nil {
		srv.modelWeights = checkpoint.Weights
		srv.totalSize = int32(len(checkpoint.Weights))
		srv.round = checkpoint.Round
		log.Printf("Resuming from checkpoint round: %d", checkpoint.Round)
	}
	srv.weightCond = sync.NewCond(&srv.mtx)
	srv.nWorkers = cfg.Server.Workers
	srv.CheckpointInterval = cfg.Server.CheckpointInterval
	srv.CheckpointPath = cfg.Server.CheckpointPath
	srv.cond = sync.NewCond(&srv.mtx)
	pb.RegisterParameterServerServer(s, srv)
	log.Printf("Parameter server listening on : %s\n", PORT)
	s.Serve(listen)
}
