#include "DQMOffline/RecoB/plugins/MiniAODTaggerAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"



MiniAODTaggerAnalyzer::MiniAODTaggerAnalyzer(const edm::ParameterSet& pSet)
: jetToken_(consumes<std::vector <pat::Jet> >(pSet.getParameter<edm::InputTag>("JetTag"))),
disrParameters_(pSet.getParameter<edm::ParameterSet>("parameters")),

folder_(pSet.getParameter<std::string>("folder")),
discrNumerator_(pSet.getParameter<vstring>("numerator")),
discrDenominator_(pSet.getParameter<vstring>("denominator")),

doCTagPlots_(pSet.getParameter<bool>("CTagPlots")),
dodifferentialPlots_(pSet.getParameter<bool>("differentialPlots")),
discrCut_(pSet.getParameter<double>("discrCut")),

etaActive_(pSet.getParameter<bool>("etaActive")),
etaMin_(pSet.getParameter<double>("etaMin")),
etaMax_(pSet.getParameter<double>("etaMax")),
ptActive_(pSet.getParameter<bool>("ptActive")),
ptMin_(pSet.getParameter<double>("ptMin")),
ptMax_(pSet.getParameter<double>("ptMax"))

{

}


MiniAODTaggerAnalyzer::~MiniAODTaggerAnalyzer() { }


void MiniAODTaggerAnalyzer::bookHistograms(DQMStore::IBooker& ibook,
                                           edm::Run const& run,
                                           edm::EventSetup const& es)
{
    jetTagPlotter_ = std::make_unique<JetTagPlotter>(folder_,
                                                     EtaPtBin(etaActive_, etaMin_, etaMax_, ptActive_, ptMin_, ptMax_),
                                                     disrParameters_,
                                                     0, //TODO MC + jetflavour (see below)
                                                     false,
                                                     ibook,
                                                     doCTagPlots_,
                                                     dodifferentialPlots_,
                                                     discrCut_
                                                    );
}


void MiniAODTaggerAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    edm::Handle<std::vector <pat::Jet> > jetCollection;
    iEvent.getByToken(jetToken_, jetCollection);

    const float jec = 1.; // JEC not implemented!
    float numerator = 0;
    float denominator = 0;

    // Loop over the pat::Jets
    for(std::vector<pat::Jet>::const_iterator jet = jetCollection->begin(); jet != jetCollection->end(); ++jet)
    {
        numerator = 0;
        denominator = 0;

        for(const auto &discrLabel: discrNumerator_)
        {
            numerator = numerator + jet->bDiscriminator(discrLabel);
        }

        for(const auto &discrLabel: discrDenominator_)
        {
            denominator = denominator + jet->bDiscriminator(discrLabel);
        }

        if(discrDenominator_.size() == 0)
        {
            denominator = 1;
        }

        // check Pt/Eta bin
        reco::Jet recoJet = *jet;
        if(jetTagPlotter_->etaPtBin().inBin(recoJet, jec))
        {
            jetTagPlotter_->analyzeTag(recoJet, jec, numerator/denominator, 0); //TODO jetflavour
        }
    }
}


//define this as a plug-in
DEFINE_FWK_MODULE(MiniAODTaggerAnalyzer);
