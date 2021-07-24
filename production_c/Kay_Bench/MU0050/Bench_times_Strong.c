void Bench_times_Strong(){
	TCanvas *c1 = new TCanvas("Weak Scaling");
//	c1->SetLogx();
//	c1->SetLogy();

//Only the unstarred columns get extracted
	TGraph *c_strong_s16_t32= new TGraph("Strong.csv", "%*lg %*lg %*lg %*lg %*lg %*lg %lg %*lg %*lg %*lg %lg","\t ,");

	c_strong_s16_t32->SetTitle("C strong Scaling, 16^{3} #times 32 point lattice, #mu=0.5 j_{qq}=0.04");
	c_strong_s16_t32->SetLineColor(kOrange+3);
	c_strong_s16_t32->SetLineStyle(2);
	c_strong_s16_t32->SetLineWidth(2);
	c_strong_s16_t32->SetMarkerColor(kOrange+3);
	c_strong_s16_t32->SetMarkerStyle(41);
	c_strong_s16_t32->SetMarkerSize(5);
	//c_strong_s16_t32->GetXaxis()->SetTitle("Lattice Size");
	c_strong_s16_t32->GetXaxis()->SetTitle("Cores");
	c_strong_s16_t32->GetYaxis()->SetTitle("Time per Trajectory (Seconds)");

	TLegend *legend1 = new TLegend();
	legend1->AddEntry(c_strong_s16_t32);

	c_strong_s16_t32->Draw();
	legend1->Draw("same");
}
