#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Common/interface/Handle.h"

#include "SimTracker/TrackTriggerAssociation/interface/StubAssociation.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"
#include "L1Trigger/TrackerTFP/interface/LayerEncoding.h"
#include "L1Trigger/TrackerTFP/interface/KalmanFilterFormats.h"

#include <TProfile.h>
#include <TH1F.h>
#include <TEfficiency.h>

#include <vector>
#include <deque>
#include <set>
#include <cmath>
#include <numeric>
#include <sstream>

using namespace std;
using namespace edm;
using namespace tt;

namespace trackerTFP {

  /*! \class  trackerTFP::AnalyzerKF
   *  \brief  Class to analyze hardware like structured TTTrack Collection generated by Kalman Filter
   *  \author Thomas Schuh
   *  \date   2020, Sep
   */
  class AnalyzerKF : public one::EDAnalyzer<one::WatchRuns, one::SharedResources> {
  public:
    AnalyzerKF(const ParameterSet& iConfig);
    void beginJob() override {}
    void beginRun(const Run& iEvent, const EventSetup& iSetup) override;
    void analyze(const Event& iEvent, const EventSetup& iSetup) override;
    void endRun(const Run& iEvent, const EventSetup& iSetup) override {}
    void endJob() override;

  private:
    //
    void associate(const TTTracks& ttTracks,
                   const StubAssociation* ass,
                   set<TPPtr>& tps,
                   int& sum,
                   const vector<TH1F*>& his,
                   TProfile* prof) const;

    // ED input token of accepted Tracks
    EDGetTokenT<StreamsStub> edGetTokenAcceptedStubs_;
    // ED input token of accepted Stubs
    EDGetTokenT<StreamsTrack> edGetTokenAcceptedTracks_;
    // ED input token of lost Stubs
    EDGetTokenT<StreamsStub> edGetTokenLostStubs_;
    // ED input token of lost Tracks
    EDGetTokenT<StreamsTrack> edGetTokenLostTracks_;
    // ED input token for number of accepted States
    EDGetTokenT<int> edGetTokenNumAcceptedStates_;
    // ED input token for number of lost States
    EDGetTokenT<int> edGetTokenNumLostStates_;
    // ED input token of TTStubRef to TPPtr association for tracking efficiency
    EDGetTokenT<StubAssociation> edGetTokenSelection_;
    // ED input token of TTStubRef to recontructable TPPtr association
    EDGetTokenT<StubAssociation> edGetTokenReconstructable_;
    // Setup token
    ESGetToken<Setup, SetupRcd> esGetTokenSetup_;
    // DataFormats token
    ESGetToken<DataFormats, DataFormatsRcd> esGetTokenDataFormats_;
    // LayerEncoding token
    ESGetToken<LayerEncoding, LayerEncodingRcd> esGetTokenLayerEncoding_;
    // stores, calculates and provides run-time constants
    const Setup* setup_ = nullptr;
    //
    const DataFormats* dataFormats_ = nullptr;
    //
    const LayerEncoding* layerEncoding_ = nullptr;
    // enables analyze of TPs
    bool useMCTruth_;
    //
    int nEvents_ = 0;

    // Histograms

    TProfile* prof_;
    TProfile* profChannel_;
    TH1F* hisChannel_;
    vector<TH1F*> hisRes_;
    TProfile* profResZ0_;
    TH1F* hisEffEta_;
    TH1F* hisEffEtaTotal_;
    TEfficiency* effEta_;
    TH1F* hisEffInv2R_;
    TH1F* hisEffInv2RTotal_;
    TEfficiency* effInv2R_;
    TH1F* hisChi2_;
    TH1F* hisPhi_;

    // printout
    stringstream log_;
  };

  AnalyzerKF::AnalyzerKF(const ParameterSet& iConfig)
      : useMCTruth_(iConfig.getParameter<bool>("UseMCTruth")), hisRes_(4) {
    usesResource("TFileService");
    // book in- and output ED products
    const string& label = iConfig.getParameter<string>("LabelKF");
    const string& branchAcceptedStubs = iConfig.getParameter<string>("BranchAcceptedStubs");
    const string& branchAcceptedTracks = iConfig.getParameter<string>("BranchAcceptedTracks");
    const string& branchLostStubs = iConfig.getParameter<string>("BranchLostStubs");
    const string& branchLostTracks = iConfig.getParameter<string>("BranchLostTracks");
    edGetTokenAcceptedStubs_ = consumes<StreamsStub>(InputTag(label, branchAcceptedStubs));
    edGetTokenAcceptedTracks_ = consumes<StreamsTrack>(InputTag(label, branchAcceptedTracks));
    edGetTokenLostStubs_ = consumes<StreamsStub>(InputTag(label, branchLostStubs));
    edGetTokenLostTracks_ = consumes<StreamsTrack>(InputTag(label, branchLostTracks));
    edGetTokenNumAcceptedStates_ = consumes<int>(InputTag(label, branchAcceptedTracks));
    ;
    edGetTokenNumLostStates_ = consumes<int>(InputTag(label, branchLostTracks));
    ;
    if (useMCTruth_) {
      const auto& inputTagSelecttion = iConfig.getParameter<InputTag>("InputTagSelection");
      const auto& inputTagReconstructable = iConfig.getParameter<InputTag>("InputTagReconstructable");
      edGetTokenSelection_ = consumes<StubAssociation>(inputTagSelecttion);
      edGetTokenReconstructable_ = consumes<StubAssociation>(inputTagReconstructable);
    }
    // book ES products
    esGetTokenSetup_ = esConsumes<Setup, SetupRcd, Transition::BeginRun>();
    esGetTokenDataFormats_ = esConsumes<DataFormats, DataFormatsRcd, Transition::BeginRun>();
    esGetTokenLayerEncoding_ = esConsumes<LayerEncoding, LayerEncodingRcd, Transition::BeginRun>();
    // log config
    log_.setf(ios::fixed, ios::floatfield);
    log_.precision(4);
  }

  void AnalyzerKF::beginRun(const Run& iEvent, const EventSetup& iSetup) {
    // helper class to store configurations
    setup_ = &iSetup.getData(esGetTokenSetup_);
    dataFormats_ = &iSetup.getData(esGetTokenDataFormats_);
    layerEncoding_ = &iSetup.getData(esGetTokenLayerEncoding_);
    // book histograms
    Service<TFileService> fs;
    TFileDirectory dir;
    dir = fs->mkdir("KF");
    prof_ = dir.make<TProfile>("Counts", ";", 11, 0.5, 11.5);
    prof_->GetXaxis()->SetBinLabel(1, "Stubs");
    prof_->GetXaxis()->SetBinLabel(2, "Tracks");
    prof_->GetXaxis()->SetBinLabel(3, "Lost Tracks");
    prof_->GetXaxis()->SetBinLabel(4, "Matched Tracks");
    prof_->GetXaxis()->SetBinLabel(5, "All Tracks");
    prof_->GetXaxis()->SetBinLabel(6, "Found TPs");
    prof_->GetXaxis()->SetBinLabel(7, "Found selected TPs");
    prof_->GetXaxis()->SetBinLabel(8, "Lost TPs");
    prof_->GetXaxis()->SetBinLabel(9, "All TPs");
    prof_->GetXaxis()->SetBinLabel(10, "states");
    prof_->GetXaxis()->SetBinLabel(11, "lost states");
    // channel occupancy
    constexpr int maxOcc = 180;
    const int numChannels = dataFormats_->numChannel(Process::kf);
    hisChannel_ = dir.make<TH1F>("His Channel Occupancy", ";", maxOcc, -.5, maxOcc - .5);
    profChannel_ = dir.make<TProfile>("Prof Channel Occupancy", ";", numChannels, -.5, numChannels - .5);
    // resoultions
    static const vector<string> names = {"phiT", "inv2R", "zT", "cot"};
    static const vector<double> ranges = {.01, .1, 5, .1};
    for (int i = 0; i < 4; i++) {
      const double range = ranges[i];
      hisRes_[i] = dir.make<TH1F>(("HisRes" + names[i]).c_str(), ";", 100, -range, range);
    }
    profResZ0_ = dir.make<TProfile>("ProfResZ0", ";", 32, 0, 2.5);
    // Efficiencies
    hisEffEtaTotal_ = dir.make<TH1F>("HisTPEtaTotal", ";", 128, -2.5, 2.5);
    hisEffEta_ = dir.make<TH1F>("HisTPEta", ";", 128, -2.5, 2.5);
    effEta_ = dir.make<TEfficiency>("EffEta", ";", 128, -2.5, 2.5);
    const double rangeInv2R = dataFormats_->format(Variable::inv2R, Process::dr).range();
    hisEffInv2R_ = dir.make<TH1F>("HisTPInv2R", ";", 32, -rangeInv2R / 2., rangeInv2R / 2.);
    hisEffInv2RTotal_ = dir.make<TH1F>("HisTPInv2RTotal", ";", 32, -rangeInv2R / 2., rangeInv2R / 2.);
    effInv2R_ = dir.make<TEfficiency>("EffInv2R", ";", 32, -rangeInv2R / 2., rangeInv2R / 2.);
    // chi2
    hisChi2_ = dir.make<TH1F>("HisChi2", ";", 100, -.5, 99.5);
    const double rangePhi = dataFormats_->format(Variable::phi0, Process::dr).range();
    hisPhi_ = dir.make<TH1F>("HisPhi", ";", 100, -rangePhi, rangePhi);
  }

  void AnalyzerKF::analyze(const Event& iEvent, const EventSetup& iSetup) {
    auto fill = [this](const TPPtr& tpPtr, TH1F* hisEta, TH1F* hisInv2R) {
      hisEta->Fill(tpPtr->eta());
      hisInv2R->Fill(tpPtr->charge() / tpPtr->pt() * setup_->invPtToDphi());
    };
    // read in kf products
    Handle<StreamsStub> handleAcceptedStubs;
    iEvent.getByToken<StreamsStub>(edGetTokenAcceptedStubs_, handleAcceptedStubs);
    const StreamsStub& acceptedStubs = *handleAcceptedStubs;
    Handle<StreamsTrack> handleAcceptedTracks;
    iEvent.getByToken<StreamsTrack>(edGetTokenAcceptedTracks_, handleAcceptedTracks);
    Handle<StreamsStub> handleLostStubs;
    iEvent.getByToken<StreamsStub>(edGetTokenLostStubs_, handleLostStubs);
    const StreamsStub& lostStubs = *handleLostStubs;
    Handle<StreamsTrack> handleLostTracks;
    iEvent.getByToken<StreamsTrack>(edGetTokenLostTracks_, handleLostTracks);
    Handle<int> handleNumAcceptedStates;
    iEvent.getByToken<int>(edGetTokenNumAcceptedStates_, handleNumAcceptedStates);
    Handle<int> handleNumLostStates;
    iEvent.getByToken<int>(edGetTokenNumLostStates_, handleNumLostStates);
    // read in MCTruth
    const StubAssociation* selection = nullptr;
    const StubAssociation* reconstructable = nullptr;
    if (useMCTruth_) {
      Handle<StubAssociation> handleSelection;
      iEvent.getByToken<StubAssociation>(edGetTokenSelection_, handleSelection);
      selection = handleSelection.product();
      prof_->Fill(9, selection->numTPs());
      Handle<StubAssociation> handleReconstructable;
      iEvent.getByToken<StubAssociation>(edGetTokenReconstructable_, handleReconstructable);
      reconstructable = handleReconstructable.product();
      for (const auto& p : selection->getTrackingParticleToTTStubsMap())
        fill(p.first, hisEffEtaTotal_, hisEffInv2RTotal_);
    }
    // analyze kf products and associate found tracks with reconstrucable TrackingParticles
    set<TPPtr> tpPtrs;
    set<TPPtr> tpPtrsSelection;
    set<TPPtr> tpPtrsLost;
    int allMatched(0);
    int allTracks(0);
    auto consume = [this](const StreamTrack& tracks, const StreamsStub& streams, int channel, TTTracks& ttTracks) {
      const int offset = channel * setup_->numLayers();
      int pos(0);
      for (const FrameTrack& frameTrack : tracks) {
        vector<StubKF> stubs;
        stubs.reserve(setup_->numLayers());
        for (int layer = 0; layer < setup_->numLayers(); layer++) {
          const FrameStub& frameStub = streams[offset + layer][pos];
          if (frameStub.first.isNonnull())
            stubs.emplace_back(frameStub, dataFormats_, layer);
        }
        TrackKF track(frameTrack, dataFormats_);
        ttTracks.emplace_back(track.ttTrack(stubs));
        pos++;
      }
    };
    for (int region = 0; region < setup_->numRegions(); region++) {
      int nStubsRegion(0);
      int nTracksRegion(0);
      int nLostRegion(0);
      for (int channel = 0; channel < dataFormats_->numChannel(Process::kf); channel++) {
        const int index = region * dataFormats_->numChannel(Process::kf) + channel;
        const StreamTrack& accepted = handleAcceptedTracks->at(index);
        const StreamTrack& lost = handleLostTracks->at(index);
        hisChannel_->Fill(accepted.size());
        profChannel_->Fill(channel, accepted.size());
        TTTracks tracks;
        const int nTracks = accumulate(accepted.begin(), accepted.end(), 0, [](int sum, const FrameTrack& frame) {
          return sum + ( frame.first.isNonnull() ? 1 : 0 );
        });
        nTracksRegion += nTracks;
        tracks.reserve(nTracks);
        consume(accepted, acceptedStubs, index, tracks);
        for (const TTTrack<Ref_Phase2TrackerDigi_>& ttTrack : tracks)
          hisPhi_->Fill(ttTrack.momentum().phi());
        nStubsRegion += accumulate(tracks.begin(), tracks.end(), 0, [](int sum, const auto& ttTrack) {
          return sum + (int)ttTrack.getStubRefs().size();
        });
        TTTracks tracksLost;
        const int nLost = accumulate(lost.begin(), lost.end(), 0, [](int sum, const FrameTrack& frame) {
          return sum + ( frame.first.isNonnull() ? 1 : 0 );
        });
        nLostRegion += nLost;
        tracksLost.reserve(nLost);
        consume(lost, lostStubs, index, tracksLost);
        allTracks += nTracks;
        if (!useMCTruth_)
          continue;
        int tmp(0);
        associate(tracks, selection, tpPtrsSelection, tmp, hisRes_, profResZ0_);
        associate(tracksLost, selection, tpPtrsLost, tmp, vector<TH1F*>(), nullptr);
        associate(tracks, reconstructable, tpPtrs, allMatched, vector<TH1F*>(), nullptr);
      }
      prof_->Fill(1, nStubsRegion);
      prof_->Fill(2, nTracksRegion);
      prof_->Fill(3, nLostRegion);
    }
    for (const TPPtr& tpPtr : tpPtrsSelection)
      fill(tpPtr, hisEffEta_, hisEffInv2R_);
    deque<TPPtr> tpPtrsRealLost;
    set_difference(tpPtrsLost.begin(), tpPtrsLost.end(), tpPtrs.begin(), tpPtrs.end(), back_inserter(tpPtrsRealLost));
    prof_->Fill(4, allMatched);
    prof_->Fill(5, allTracks);
    prof_->Fill(6, tpPtrs.size());
    prof_->Fill(7, tpPtrsSelection.size());
    prof_->Fill(8, tpPtrsRealLost.size());
    prof_->Fill(10, *handleNumAcceptedStates);
    prof_->Fill(11, *handleNumLostStates);
    nEvents_++;
  }

  void AnalyzerKF::endJob() {
    if (nEvents_ == 0)
      return;
    // effi
    effEta_->SetPassedHistogram(*hisEffEta_, "f");
    effEta_->SetTotalHistogram(*hisEffEtaTotal_, "f");
    effInv2R_->SetPassedHistogram(*hisEffInv2R_, "f");
    effInv2R_->SetTotalHistogram(*hisEffInv2RTotal_, "f");
    // printout SF summary
    const double totalTPs = prof_->GetBinContent(9);
    const double numStubs = prof_->GetBinContent(1);
    const double numTracks = prof_->GetBinContent(2);
    const double numTracksLost = prof_->GetBinContent(3);
    const double totalTracks = prof_->GetBinContent(5);
    const double numTracksMatched = prof_->GetBinContent(4);
    const double numTPsAll = prof_->GetBinContent(6);
    const double numTPsEff = prof_->GetBinContent(7);
    const double numTPsLost = prof_->GetBinContent(8);
    const double errStubs = prof_->GetBinError(1);
    const double errTracks = prof_->GetBinError(2);
    const double errTracksLost = prof_->GetBinError(3);
    const double fracFake = (totalTracks - numTracksMatched) / totalTracks;
    const double fracDup = (numTracksMatched - numTPsAll) / totalTracks;
    const double eff = numTPsEff / totalTPs;
    const double errEff = sqrt(eff * (1. - eff) / totalTPs / nEvents_);
    const double effLoss = numTPsLost / totalTPs;
    const double errEffLoss = sqrt(effLoss * (1. - effLoss) / totalTPs / nEvents_);
    const int numStates = prof_->GetBinContent(10);
    const int numStatesLost = prof_->GetBinContent(11);
    const double fracSatest = numStates / (double)(numStates + numStatesLost);
    const vector<double> nums = {numStubs, numTracks, numTracksLost};
    const vector<double> errs = {errStubs, errTracks, errTracksLost};
    const int wNums = ceil(log10(*max_element(nums.begin(), nums.end()))) + 5;
    const int wErrs = ceil(log10(*max_element(errs.begin(), errs.end()))) + 5;
    log_ << "                         KF  SUMMARY                         " << endl;
    log_ << "number of stubs       per TFP = " << setw(wNums) << numStubs << " +- " << setw(wErrs) << errStubs << endl;
    log_ << "number of tracks      per TFP = " << setw(wNums) << numTracks << " +- " << setw(wErrs) << errTracks
         << endl;
    log_ << "number of lost tracks per TFP = " << setw(wNums) << numTracksLost << " +- " << setw(wErrs) << errTracksLost
         << endl;
    log_ << "          tracking efficiency = " << setw(wNums) << eff << " +- " << setw(wErrs) << errEff << endl;
    log_ << "     lost tracking efficiency = " << setw(wNums) << effLoss << " +- " << setw(wErrs) << errEffLoss << endl;
    log_ << "                    fake rate = " << setw(wNums) << fracFake << endl;
    log_ << "               duplicate rate = " << setw(wNums) << fracDup << endl;
    log_ << "    state assessment fraction = " << setw(wNums) << fracSatest << endl;
    log_ << "=============================================================";
    LogPrint("L1Trigger/TrackerTFP") << log_.str();
  }

  //
  void AnalyzerKF::associate(const TTTracks& ttTracks,
                             const StubAssociation* ass,
                             set<TPPtr>& tps,
                             int& sum,
                             const vector<TH1F*>& his,
                             TProfile* prof) const {
    for (const TTTrack<Ref_Phase2TrackerDigi_>& ttTrack : ttTracks) {
      const vector<TTStubRef>& ttStubRefs = ttTrack.getStubRefs();
      const vector<TPPtr>& tpPtrs = ass->associateFinal(ttStubRefs);
      if (tpPtrs.empty())
        continue;
      sum++;
      copy(tpPtrs.begin(), tpPtrs.end(), inserter(tps, tps.begin()));
      if (his.empty())
        continue;
      for (const TPPtr& tpPtr : tpPtrs) {
        const double phi0 = tpPtr->phi();
        const double cot = sinh(tpPtr->eta());
        const double inv2R = setup_->invPtToDphi() * tpPtr->charge() / tpPtr->pt();
        const math::XYZPointD& v = tpPtr->vertex();
        const double z0 = v.z() - cot * (v.x() * cos(phi0) + v.y() * sin(phi0));
        const double dCot = cot - ttTrack.tanL();
        const double dZ0 = z0 - ttTrack.z0();
        const double dInv2R = inv2R - ttTrack.rInv();
        const double dPhi0 = deltaPhi(phi0 - ttTrack.phi());
        const vector<double> ds = {dPhi0, dInv2R, dZ0, dCot};
        for (int i = 0; i < (int)ds.size(); i++)
          his[i]->Fill(ds[i]);
        prof->Fill(abs(tpPtr->eta()), abs(dZ0));
      }
    }
  }

}  // namespace trackerTFP

DEFINE_FWK_MODULE(trackerTFP::AnalyzerKF);
