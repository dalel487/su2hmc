void Bench_times(){
	TCanvas *c1 = new TCanvas("Weak Scaling");
//	c1->SetLogx(); c1->SetLogy();

//Only the unstarred columns get extracted
	TGraph *c_weak_256= new TGraph("C_Weak_256.csv", "%*lg %*lg %*lg %*lg %*lg %*lg %lg %*lg %*lg %*lg %lg","\t ,");
	TGraph *Fort_weak_256= new TGraph("Fort_Weak_256.csv", "%*lg %*lg %*lg %*lg %*lg %*lg %lg %*lg %*lg %*lg %lg","\t ,");

	c_weak_256->SetTitle("C Weak Scaling, 256 point sub lattice, 75-85\% Acceptance");
	c_weak_256->SetLineColor(kOrange+3);
	c_weak_256->SetLineStyle(2);
	c_weak_256->SetLineWidth(2);
	c_weak_256->SetMarkerColor(kOrange+3);
	c_weak_256->SetMarkerStyle(41);
	c_weak_256->SetMarkerSize(5);
	//c_weak_256->GetXaxis()->SetTitle("Lattice Size");
	c_weak_256->GetXaxis()->SetTitle("Cores");
	c_weak_256->GetYaxis()->SetTitle("Time per Trajectory (Seconds)");

	TLegend *legend1 = new TLegend();
	legend1->AddEntry(c_weak_256);

	Fort_weak_256->SetTitle("Fortran Weak Scaling, 256 point sub lattice, 75-85\% Acceptance");
	Fort_weak_256->SetLineColor(kAzure+7);
	Fort_weak_256->SetLineStyle(4);
	Fort_weak_256->SetLineWidth(2);
	Fort_weak_256->SetMarkerColor(kAzure+7);
	Fort_weak_256->SetMarkerStyle(23);
	Fort_weak_256->SetMarkerSize(5);

	//Fort_weak_256->GetXaxis()->SetTitle("Lattice Size");
	Fort_weak_256->GetXaxis()->SetTitle("Cores");
	Fort_weak_256->GetYaxis()->SetTitle("Time per Trajectory (Seconds)");
	legend1->AddEntry(Fort_weak_256);

	Fort_weak_256->Draw();
	c_weak_256->Draw("same");
	legend1->Draw("same");
}
